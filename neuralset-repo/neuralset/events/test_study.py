# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import typing as tp
import warnings
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import exca
import numpy as np
import pandas as pd
import pydantic
import pytest
from exca.steps.conftest import extract_cache_folders

import neuralset as ns
from neuralset import events as _events
from neuralset import segments
from neuralset.events import etypes, transforms
from neuralset.events import study as base

from .transforms import test_transforms

logging.getLogger("exca").setLevel(logging.WARNING)


def _check_loaded(step: ns.Step) -> bool:
    """Run a step in a subprocess and verify STUDY_PATHS is populated."""
    if "fork" not in mp.get_start_method():
        msg = "non-forked subprocess should start with empty STUDY_PATHS"
        assert not base.STUDY_PATHS, msg
    _ = step.run()
    if isinstance(step, ns.Chain):
        step = next(iter(step._step_sequence()))  # type: ignore
    return type(step).__name__ in base.STUDY_PATHS


class DoNothing(transforms.EventsTransform):
    param: str
    is_default: int = 12

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        return events


@pytest.mark.sandbox_skip
def test_chain_with_transform(tmp_path: Path) -> None:
    steps: tp.Any = [
        {"name": "Mne2013Sample", "path": ns.CACHE_FOLDER},
        {"name": "DoNothing", "param": "blublu-param"},
    ]
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "force"}
    chain = ns.Chain(steps=steps, infra=infra)
    df = chain.run()
    assert isinstance(df, pd.DataFrame)
    uid = exca.ConfDict.from_model(chain, uid=True, exclude_defaults=True).to_uid()
    assert "DoNothing" in uid
    assert "blublu-param" in uid
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        job = ex.submit(_check_loaded, chain)
    assert job.result(), "Instances should have been registered at load time"


def test_study_direct_instantiation_blocked(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        base.Study(path=tmp_path)


def test_study_name_collision() -> None:
    """__init_subclass__ rejects a study whose name is already registered.

    Temporarily injects a sentinel into STUDIES instead of creating a real
    Study subclass, because a failed __init_subclass__ still leaves the
    class in __subclasses__(), which confuses exca's discriminated dispatch.
    """
    sentinel = type("Sentinel", (), {})
    base.STUDIES["CollisionProbe9999"] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="Study name collision"):

            class CollisionProbe9999(base.Study):  # pylint: disable=unused-variable
                pass

    finally:
        base.STUDIES.pop("CollisionProbe9999", None)


def test_study_dispatch_by_name(tmp_path: Path) -> None:
    study = base.Study(name="Allen2022Massive", path=tmp_path)  # type: ignore[call-arg]
    assert type(study).__name__ == "Allen2022Massive"
    assert isinstance(study, base.Study)


def test_loader_on_external_study(tmp_path: Path) -> None:
    study = ns.Study(name="FakeData2025", path=tmp_path)
    assert isinstance(study, FakeData2025)
    with pytest.raises(ImportError, match="not found"):
        ns.Study(name="FakeData2223", path=tmp_path)


def test_study_download(tmp_path: Path) -> None:
    infra: tp.Any = {"cluster": None}
    study = FakeData2025(path=tmp_path, infra_timelines=infra)
    study._download = lambda: (study.path / "data.txt").touch()  # type: ignore
    study.download()
    assert study.path.exists()
    assert (study.path / "data.txt").is_file()
    assert oct(os.stat(study.path).st_mode & 0o777) == "0o777"


@pytest.mark.parametrize("with_name", [False, True])
def test_download_appends_study_name(tmp_path: Path, with_name: bool) -> None:
    path = tmp_path / "FakeData2025" if with_name else tmp_path
    infra: tp.Any = {"cluster": None}
    study = FakeData2025(path=path, infra_timelines=infra)
    study._download = lambda: (study.path / "data.txt").touch()  # type: ignore
    study.download()
    assert study.path == tmp_path / "FakeData2025"
    assert (study.path / "data.txt").is_file()


def test_study_export() -> None:
    study = ns.Study(name="Mne2013Sample", path=ns.CACHE_FOLDER)
    dumped = study.model_dump()
    ns.Study(**dumped)


def test_study_summary_and_build(tmp_path: Path) -> None:
    study = FakeData2025(path=tmp_path, query="subject_index < 1")
    summary = study.study_summary()
    assert summary.subject.nunique() == 1
    assert len(summary) == 2
    summary_no_query = study.study_summary(apply_query=False)
    assert summary_no_query.subject.nunique() == 2
    assert len(summary_no_query) == 4
    out = study.run()
    assert "limb" in out.columns
    segs = segments.list_segments(out, out.type == "Action")
    trigger = segs[0].trigger
    assert isinstance(trigger, etypes.Action)
    assert "limb" in trigger.extra
    # back and forth
    events = [ns.events.Event.from_dict(d) for d in out.itertuples(index=False)]
    out2 = pd.DataFrame([e.to_dict() for e in events], columns=out.columns)
    pd.testing.assert_frame_equal(out2, out)
    # timelines
    assert set(summary.timeline) == set(out.timeline)


def test_study_summary_sorts_by_subject_before_query(tmp_path: Path) -> None:
    """FakeData2025 iterates run×subject (interleaved: subj12_run0, subj13_run0,
    subj12_run1, subj13_run1). After stable-sorting by subject, subject 12's
    timelines come first, so timeline_index < 2 must return both timelines
    of subject 12 — not one from each subject.
    """
    study = FakeData2025(
        path=tmp_path / "data",
        query="timeline_index < 2",
    )
    summ = study.study_summary(apply_query=True)
    assert len(summ) == 2
    assert summ.subject.nunique() == 1
    assert summ.subject.iloc[0] == "FakeData2025/12"
    # Without query, all 4 timelines are present and sorted by subject
    summ_all = study.study_summary(apply_query=False)
    assert len(summ_all) == 4
    subjects = summ_all.subject.tolist()
    assert subjects == sorted(subjects)


class Xp(ns.BaseModel):
    study: ns.Step
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def build(self) -> pd.DataFrame:
        return self.study.run()


def test_xp(tmp_path: Path) -> None:
    xp = Xp(
        infra={"folder": tmp_path / "xp", "cluster": "local"},  # type: ignore
        study={  # type: ignore[arg-type]
            "name": "Mne2013Sample",
            "path": ns.CACHE_FOLDER,
            "infra_timelines": {"cluster": None},
            "infra": {"backend": "Cached", "folder": tmp_path / "study"},
        },
    )
    # test pickling
    string = cloudpickle.dumps(xp)
    unpickled = cloudpickle.loads(string)
    assert type(unpickled.study).__name__ == type(xp.study).__name__
    assert "path" in xp.study._exclude_from_cls_uid()
    # run it
    xp.build()
    subs = list((tmp_path / "study").iterdir())
    assert len(subs) >= 1, "Expected at least one cache folder"


### STUDY V2 ###


class FakeData2025(base.Study):
    # study level
    myparam: int = 12
    # class level

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # sequential: no need to spawn processes for test data
        self.infra_timelines.cluster = None

    # for testing special loader param
    _add_offset: float = 0

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for run in range(2):
            for subject in [12, 13]:
                yield {"subject": subject, "run": run, "task": "motor"}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        kwargs = {}
        if self._add_offset:
            kwargs = {"offset": self._add_offset}
        loader = base.SpecialLoader(method=self._get_img, timeline=timeline, **kwargs)
        motor = {
            "type": "Action",
            "start": 0,
            "duration": 12,
            "limb": "leg",
            "stuff": True,
            "blublu": timeline["run"] + timeline["subject"],
        }
        sound = {
            "type": "Image",
            "start": 0,
            "duration": 12,
            "filepath": loader.to_json(),
        }
        return pd.DataFrame([motor, sound])

    # pylint: disable=unused-argument
    def _get_img(self, timeline: dict[str, tp.Any], offset: float = 0) -> tp.Any:
        return np.zeros((12, 12)) + offset


@pytest.mark.parametrize(
    "d, expected",
    [
        ({"run": 0, "subject": 12, "task": "motor"}, "run=0,subject=12,task=motor"),
        ({"run": None, "session": "9"}, "run=None,session=9"),
        ({}, ""),
        ({"a": 1}, "a=1"),
    ],
)
def test_dict_to_kv(d: dict[str, tp.Any], expected: str) -> None:
    assert base._dict_to_kv(d) == expected


def test_timeline_string_pandas_query(tmp_path: Path) -> None:
    """Timeline strings must be usable in pd.DataFrame.query() without escaping."""
    infra: tp.Any = {"folder": tmp_path, "cluster": None}
    study = FakeData2025(path=tmp_path / "data", infra_timelines=infra)
    df = study.run()
    assert df.timeline.nunique() == 4
    tl = df.timeline.iloc[0]
    filtered = df.query("timeline == @tl")
    assert len(filtered) == 2
    assert (filtered.timeline == tl).all()


def test_study_build(tmp_path: Path) -> None:
    """Loading, columns, BIDS entities, summary, event round-trip."""
    study = FakeData2025(path=tmp_path / "data")
    tls = list(study.iter_timelines())
    assert len(tls) == 4
    df = study.run()
    assert "study" in df.columns
    assert df.loc[0, "study"] == "FakeData2025"
    assert len(df) == 8
    assert df.timeline[0] == "FakeData2025:run=0,subject=12,task=motor"
    # BIDS entity columns are present, correctly typed and populated
    for entity in _events.utils.BIDS_ENTITIES:
        assert entity in df.columns
        assert df[entity].dtype == object  # str
    assert df.loc[0, "task"] == "motor"
    assert df.loc[0, "session"] == _events.utils.BIDS_ENTITY_DEFAULT
    summ = study.study_summary()
    assert summ.timeline[0] == df.timeline[0]
    # SpecialLoader round-trip
    image = etypes.Image.from_dict(next(iter(df.loc[df.type == "Image"].itertuples())))
    s = '{"cls":"FakeData2025","method":"_get_img","timeline":{"run":0,"subject":12,"task":"motor"}}'
    assert image.filepath == s
    im = image.read()
    assert im.shape == (12, 12)
    assert not im[0, 0]
    assert not study.requirements


def test_study_caching(tmp_path: Path) -> None:
    """Per-timeline caching, query filtering, infra.folder propagation."""
    # infra.folder propagates to infra_timelines.folder when unset
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path, "mode": "force"}
    study = FakeData2025(path=tmp_path / "data", infra=infra)
    assert study.infra_timelines.folder == tmp_path
    assert study.infra_timelines.mode == "force"
    # retry mode propagates as cached
    assert study.infra is not None
    infra_retry = study.infra.model_copy(update={"mode": "retry"}, deep=True)
    study_retry = FakeData2025(path=tmp_path / "data", infra=infra_retry)
    assert study_retry.infra_timelines.mode == "cached"
    # explicit folder: build and check cache files
    infra_timelines: tp.Any = {"folder": tmp_path, "cluster": None}
    study = FakeData2025(path=tmp_path / "data", infra_timelines=infra_timelines)
    study.run()
    folder = study.infra_timelines.uid_folder()
    assert folder is not None
    fps = list((folder / "data").glob("*.parquet"))
    assert len(fps) == 4
    # query filters timelines but does not create new cache files
    study = FakeData2025(
        path=tmp_path / "data", infra_timelines=infra_timelines, query="run==1"
    )
    df = study.run()
    assert len(df) == 4
    assert len(list((folder / "data").glob("*.parquet"))) == 4


class _SimpleStudy(base.Study):
    """Minimal study for testing timeline dict handling and BIDS defaults."""

    _timelines: list[dict[str, tp.Any]] = [
        {"subject": "s1", "run": 0, "condition": "rest", "block": 3}
    ]
    _event_condition: str | None = None

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        yield from self._timelines

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # timeline/subject are overwritten by the framework (tested by test_timeline_dict_auto_columns)
        event: dict[str, tp.Any] = {
            "type": "Motor",
            "start": 0,
            "duration": 1,
            "timeline": "BOGUS",
            "subject": "BOGUS",
        }
        if self._event_condition is not None:
            event["condition"] = self._event_condition
        return pd.DataFrame([event])


def _build_simple(tmp_path: Path, **overrides: tp.Any) -> pd.DataFrame:
    infra: tp.Any = {"folder": tmp_path, "cluster": None}
    study = _SimpleStudy(path=tmp_path / "data", infra_timelines=infra)
    for k, v in overrides.items():
        setattr(study, k, v)
    return study.run()


def test_timeline_dict_auto_columns(tmp_path: Path) -> None:
    df = _build_simple(tmp_path)
    assert df.loc[0, "condition"] == "rest"
    assert str(df.loc[0, "block"]) == "3"
    # BIDS entities default to BIDS_ENTITY_DEFAULT when not in timeline dict
    assert df.loc[0, "task"] == _events.utils.BIDS_ENTITY_DEFAULT
    assert df.loc[0, "session"] == _events.utils.BIDS_ENTITY_DEFAULT
    assert df.loc[0, "run"] == "0"  # run was provided as int, cast to str


def test_timeline_dict_conflict_matching(tmp_path: Path) -> None:
    tl = [{"subject": "s1", "condition": "rest"}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _build_simple(tmp_path, _timelines=tl, _event_condition="rest")
    future_warns = [x for x in w if issubclass(x.category, FutureWarning)]
    assert any("condition" in str(x.message) for x in future_warns)


def test_timeline_dict_conflict_different(tmp_path: Path) -> None:
    tl = [{"subject": "s1", "condition": "rest"}]
    with pytest.raises(ValueError, match="condition.*different values"):
        _build_simple(tmp_path, _timelines=tl, _event_condition="active")


def test_study_v2_param(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path}
    study = FakeData2025(path=tmp_path / "data", infra=infra, myparam=13)
    with pytest.raises(RuntimeError):
        _ = study.run()
    # disabled for now:
    # assert len(df) == 8
    # summ = study.study_summary()
    # expected = "FakeData2025:myparam=13:run=0,subject=12"
    # assert summ.timeline[0] == expected
    # assert df.timeline[0] == expected


# # # # # Study chain # # # # #


def test_study_chain(tmp_path: Path) -> None:
    chain = test_transforms._custom_study(tmp_path, tmp_path)
    assert isinstance(chain, ns.Chain)
    events = chain.run()
    events = events.query('type=="Audio"')
    assert len(events) == 6
    assert events.duration.max() == 5.0
    cache_folders = extract_cache_folders(tmp_path / "cache")
    assert len(cache_folders) >= 1


@pytest.mark.parametrize("as_dict", (True, False))
def test_raw_studychain(tmp_path: Path, as_dict: bool) -> None:
    _ = test_transforms._custom_study(tmp_path, tmp_path)  # creates wav data
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps: tp.Any = [
        {
            "name": "Test2023Meg",
            "path": tmp_path / "data",
        },
        {"name": "FakeChapter", "path": tmp_path, "infra": infra},  # cache
        {"name": "ChunkEvents", "event_type_to_chunk": "Audio", "max_duration": 5.0},
    ]
    if as_dict:
        steps = {str(k): s for k, s in enumerate(steps)}

    chain = ns.Chain(steps=steps, infra=infra)
    events = chain.run()
    # check results
    events = events.query('type=="Audio"')
    assert len(events) == 6
    assert events.duration.max() == 5.0


class BadEnhancer(transforms.EventsTransform):
    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        # negative duration should trigger pydantic ValidationError
        events.duration = -1  # type: ignore
        return events


def test_bad_transform(tmp_path: Path):
    steps: tp.Any = [
        {
            "name": "FakeData2025",
            "path": tmp_path,
            "infra_timelines": {"cluster": None},
        },
        {"name": "BadEnhancer"},
    ]
    # Chain must inherit ValidatedParquet from its events steps
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain = ns.Chain(steps=steps, infra=infra)
    with pytest.raises(pydantic.ValidationError):
        _ = chain.run()


def test_study_discovery(tmp_path: Path) -> None:
    """Step._discover_study triggers lazy import for unknown study names.

    Covers both Chain(steps=[study_dict]) and a bare study dict in an
    ns.Step-typed pydantic field.  Runs in a subprocess so Bel2026PetitListen
    is guaranteed not to have been imported by earlier tests.
    """
    code = "\n".join(
        [
            "import pydantic",
            "import neuralset as ns",
            "from neuralset.events import study",
            "assert 'Bel2026PetitListen' not in study.STUDIES, 'already loaded'",
            # bare Step field
            "class Cfg(pydantic.BaseModel):",
            "    step: ns.Step",
            f"cfg = Cfg(step=dict(name='Bel2026PetitListen', path='{tmp_path}'))",
            "assert type(cfg.step).__name__ == 'Bel2026PetitListen'",
            # Chain wrapping
            f"c = ns.Chain(steps=[dict(name='Bel2026PetitListen', path='{tmp_path}')])",
            "assert type(c.steps[0]).__name__ == 'Bel2026PetitListen'",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_concat_studies(tmp_path: Path) -> None:
    step: tp.Any = {
        "name": "FakeData2025",
        "path": tmp_path / "data",
        "infra_timelines": {"cluster": None},
    }
    chain = ns.Chain(steps=[step])
    df = chain.run()
    assert len(df) == 8
    chain = ns.Chain(steps=[step, step])
    df = chain.run()
    assert len(df) == 16


def test_intermediary_study_cache(tmp_path: Path) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    steps: list[tp.Any] = [
        # cache after study as it can be slow
        {"name": "Fake2025Meg", "path": ns.CACHE_FOLDER, "infra": infra},
        #  Cache after chunking - useful if chunking is slow
        {
            "name": "ChunkEvents",
            "event_type_to_chunk": "Audio",
            "max_duration": 5.0,
            "infra": infra,
        },
        # Step 4: Cache after chunking - useful if chunking is slow
        {"name": "QueryEvents", "query": "type in ['Audio', 'Word', 'Meg']"},
    ]
    chain = ns.Chain(steps=steps, infra=infra)
    assert isinstance(chain.steps[0], base.Study)  # type: ignore
    _ = chain.run()
    # check reloaded even with named steps
    named: tp.Any = {str(k): s for k, s in enumerate(steps)}
    chain = ns.Chain(steps=named, infra=infra)
    assert chain.infra is not None
    df = chain.infra.cached_result()
    assert isinstance(df, pd.DataFrame)  # cached
    assert hasattr(df, "stop")  # validated
    chain_caches = set(extract_cache_folders(tmp_path))
    expected = [
        "name=Fake2025Meg,infra_timelines.version=v3-66b17a23",
        "max_duration=5,name=ChunkEvents,event_type_to_chunk=Audio-7702b0cc",
        "name=QueryEvents,query=type-in-Audio,-Word,-Meg-c3b10366",
    ]
    for k in range(len(expected) - 1):  # concatenate as each cache is a subfolder
        expected[k + 1] = expected[k] + "/" + expected[k + 1]
    assert chain_caches == set(expected)


def test_validate_events_in_steps(tmp_path: Path) -> None:
    steps: list[tp.Any] = [
        {"name": "Fake2025Meg", "path": ns.CACHE_FOLDER},
        {"name": "ChunkEvents", "event_type_to_chunk": "Audio", "max_duration": 5.0},
    ]
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    chain_process = ns.Chain(steps=steps, infra=infra)
    events = chain_process.run()
    assert hasattr(events, "stop")  # validated


def test_sub_chain(tmp_path) -> None:
    steps: tp.Any = [
        # Chunk audio into smaller pieces
        {"name": "ChunkEvents", "event_type_to_chunk": "Audio", "max_duration": 5.0},
    ]
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path / "cache"}
    audio_subchain = ns.Chain(steps=steps, infra=infra)
    # Main chain that uses the sub-chain
    steps = [
        # Step 1: Load the study
        {"name": "Fake2025Meg", "path": ns.CACHE_FOLDER},
        # Step 2: Apply audio preprocessing sub-chain
        audio_subchain,
        # Step 3: Filter to relevant events
        {"name": "QueryEvents", "query": "type in ['Audio', 'Word', 'Meg']"},
    ]
    main_chain = ns.Chain(steps=steps, infra=infra)
    events = main_chain.run()
    assert isinstance(events, pd.DataFrame)


def test_no_full_scan_on_unpickle(tmp_path: Path) -> None:
    """Unpickling a Study must not trigger a full package scan (name='')."""
    infra: tp.Any = {"cluster": None}
    study = FakeData2025(path=tmp_path / "data", infra_timelines=infra)
    data = cloudpickle.dumps(study)
    scan_calls: list[str] = []
    real_scan = base._scan_package_for_studies

    def _tracking_scan(base_dir, base_module, name):
        scan_calls.append(name)
        return real_scan(base_dir, base_module, name)

    with patch.object(base, "_scan_package_for_studies", side_effect=_tracking_scan):
        cloudpickle.loads(data)
    full_scans = [n for n in scan_calls if not n]
    assert not full_scans, (
        f"Unpickling triggered {len(full_scans)} full package scan(s); "
        f"__new__ should skip _resolve_study when name is missing"
    )


@pytest.mark.parametrize("with_files", [False, True])
def test_set_dir_permissions(tmp_path: Path, with_files: bool) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()
    if with_files:
        (subdir / "file.txt").write_text("hello")
        (subdir / "nested").mkdir()
        (subdir / "nested" / "deep.txt").write_text("world")
    base._set_dir_permissions(subdir)
    assert oct(os.stat(subdir).st_mode & 0o777) == "0o777"
    if with_files:
        assert oct(os.stat(subdir / "file.txt").st_mode & 0o777) == "0o777"
        assert oct(os.stat(subdir / "nested" / "deep.txt").st_mode & 0o777) == "0o777"
