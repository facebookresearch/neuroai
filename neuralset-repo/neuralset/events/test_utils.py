# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from exca import cachedict

from neuralset import segments
from neuralset.events import etypes, utils

from .test_study import FakeData2025


def test_extract_events(tmp_path: Path) -> None:
    study = FakeData2025(path=tmp_path, query="subject_index < 1")
    out = study.run()
    # extract from various input types: DataFrame, segments, single event, dict, list
    all_events = utils.extract_events(out)
    actions = utils.extract_events(out, types="Action")
    assert len(all_events) == len(actions) + len(utils.extract_events(out, types="Image"))
    assert len(utils.extract_events(all_events[0])) == 1
    assert len(utils.extract_events(all_events[0].to_dict())) == 1
    assert len(utils.extract_events(all_events)) == len(all_events)
    # from segments (includes trigger + neighboring events)
    segs = segments.list_segments(out, out.type == "Action")
    assert len(utils.extract_events(segs)) == len(all_events)
    # type filtering: missing type returns empty, EventTypesHelper equivalent
    assert not utils.extract_events(out, types="Stimulus")
    assert len(etypes.EventTypesHelper("Action").extract(out)) == len(actions)
    # extract from a segment built via list_segments
    seg0 = segs[0]
    seg0_events = utils.extract_events(seg0)
    assert len(seg0_events) >= 1


def test_special_loader_kwargs(tmp_path: Path) -> None:
    params: tp.Any = dict(path=tmp_path / "data", infra={"folder": tmp_path})
    study = FakeData2025(**params)
    study._add_offset = 1
    df = study.run()
    # def read data
    image = etypes.Image.from_dict(next(iter(df.loc[df.type == "Image"].itertuples())))
    s = '{"cls":"FakeData2025","kwargs":{"offset":1},"method":"_get_img","timeline":{"run":0,"subject":12,"task":"motor"}}'
    assert image.filepath == s
    assert image.read()[0, 0] == 1


def test_expand_bids_fmri(tmp_path: Path) -> None:
    """expand_bids_fmri resolves a BIDS glob into per-space Fmri dicts."""
    from .test_etypes import FMRI_SHAPE_4D, _make_cifti, _make_gifti, _make_nifti

    base = "sub-01_ses-01_task-rest_run-1_"
    _make_nifti(tmp_path, base + "space-T1w_desc-preproc_bold.nii.gz")
    _make_nifti(tmp_path, base + "space-T1w_desc-brain_mask.nii.gz", FMRI_SHAPE_4D[:-1])
    _make_gifti(tmp_path, base + "hemi-L_space-fsaverage_bold.func.gii")
    _make_gifti(tmp_path, base + "hemi-R_space-fsaverage_bold.func.gii")
    _make_cifti(tmp_path, base + "space-fsLR_den-91k_bold.dtseries.nii")

    result = utils.expand_bids_fmri(
        str(tmp_path / (base + "*")), preproc="deepprep", start=0, frequency=0.5
    )
    spaces = {r["space"] for r in result}
    assert spaces == {"T1w", "fsaverage", "fsLR"}
    vol = next(r for r in result if r["space"] == "T1w")
    assert "preproc_bold" in vol["filepath"]
    assert vol.get("mask_filepath") is not None
    surf = next(r for r in result if r["space"] == "fsaverage")
    assert "hemi-L" in surf["filepath"]
    cifti = next(r for r in result if r["space"] == "fsLR")
    assert cifti["filepath"].endswith("_bold.dtseries.nii")
    assert "mask_filepath" not in cifti
    assert cifti["spec"] == {"den": "91k"}
    assert all(r["preproc"] == "deepprep" for r in result)


def test_validated_parquet_no_nullable_dtypes(tmp_path: Path) -> None:
    """ValidatedParquet round-trip must produce np.nan, not pd.NA.

    The parent class reads with dtype_backend="numpy_nullable", turning
    float64 into Float64. On Float64, diff() produces pd.NA (not np.nan)
    and bool(pd.NA) raises TypeError — breaking any(series.diff() < 0).
    """
    events = pd.DataFrame(
        [
            dict(type="Word", start=0.0, duration=0.5, timeline="t1"),
            dict(type="Word", start=1.0, duration=0.5, timeline="t1"),
        ]
    )
    fp = tmp_path / "events.parquet"
    events.to_parquet(fp)
    ctx = cachedict.DumpContext(folder=tmp_path)
    loaded = utils.ValidatedParquet.__load_from_info__(ctx, "events.parquet")
    assert np.isnan(loaded.start.diff().iloc[0])


def test_mixed_type_column_parquet(tmp_path: Path) -> None:
    """Columns with mixed types (int + str) should serialize to parquet."""
    events = pd.DataFrame(
        [
            {
                "type": "Word",
                "text": "a",
                "start": 0,
                "duration": 1,
                "timeline": "a",
                "foo": 1,
            },
            {
                "type": "Word",
                "text": "b",
                "start": 0,
                "duration": 1,
                "timeline": "b",
                "foo": "x",
            },
        ]
    )
    cd: tp.Any = cachedict.CacheDict(folder=tmp_path)
    with cd.writer() as writer:
        writer["test"] = events
    loaded = cd["test"]
    assert set(loaded["foo"].astype(str)) == {"1", "x"}


def _stimuli(starts: tp.Sequence[float], durations: tp.Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": "Stimulus",
            "timeline": "t0",
            "start": list(starts),
            "duration": list(durations),
        }
    )


def test_warn_overlapping_durations() -> None:
    """Durations far longer than the event spacing are a unit error (ds004357)."""
    # 50 ms apart, each claiming 16 s: milliseconds stored in a seconds column
    starts = [i * 0.05 for i in range(60)]
    with pytest.warns(UserWarning, match="overlap the next same-type event"):
        utils._warn_overlapping_durations(_stimuli(starts, [16.0] * 60))

    # same spacing, honest durations: quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils._warn_overlapping_durations(_stimuli(starts, [0.016] * 60))


def test_warn_overlapping_durations_allows_tiling() -> None:
    """Contiguous fixed-length segments end as the next begins, so never overlap."""
    starts = [i * 3.0 for i in range(40)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils._warn_overlapping_durations(_stimuli(starts, [3.0] * 40))


def test_warn_whole_number_durations() -> None:
    """One whole-second duration across irregular events is a count (ds004579).

    The overlap check cannot see this case: the bogus 1 s sits just under the
    ~1.1 s event spacing, so nothing visibly overlaps.
    """
    rng = np.random.default_rng(0)
    starts = np.cumsum(rng.uniform(0.05, 4.0, size=80))
    events = _stimuli(starts, [1.0] * 80)
    with pytest.warns(UserWarning, match="whole-number duration"):
        utils._warn_whole_number_durations(events)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils._warn_overlapping_durations(events)


def test_warn_whole_number_durations_quiet_cases() -> None:
    rng = np.random.default_rng(0)
    irregular = np.cumsum(rng.uniform(0.05, 4.0, size=80))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # a count is never fractional, so a 2 ms trigger is not suspicious
        utils._warn_whole_number_durations(_stimuli(irregular, [0.002] * 80))
        # regular spacing: a block design, not a unit error
        block = [i * 6.0 for i in range(40)]
        utils._warn_whole_number_durations(_stimuli(block, [6.0] * 40))
        # a mix of durations is not a single pasted constant
        mixed = [1.0 if i % 2 else 0.002 for i in range(80)]
        utils._warn_whole_number_durations(_stimuli(irregular, mixed))


def test_check_event_durations_reports_context() -> None:
    """Callers that read a file name it, so the warning points at the source."""
    rng = np.random.default_rng(0)
    starts = np.cumsum(rng.uniform(0.05, 4.0, size=80))
    with pytest.warns(UserWarning, match="whole-number duration") as record:
        utils.check_event_durations(_stimuli(starts, [1.0] * 80), context="sub-01.tsv")
    assert "Source: sub-01.tsv" in str(record[0].message)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils.check_event_durations(_stimuli(starts, [0.25] * 80))
