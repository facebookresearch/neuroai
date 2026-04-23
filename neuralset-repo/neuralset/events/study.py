# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import importlib.metadata
import logging
import os
import typing as tp
import warnings
from pathlib import Path

import exca
import exca.steps
import pandas as pd
import pydantic
import ujson

from neuralset import base

from . import etypes, utils

# re-export: study implementations use study.SpecialLoader
from .utils import SpecialLoader as SpecialLoader  # isort: skip  # pylint: disable=W0611

logger = logging.getLogger(__name__)


def _check_folder_path(path: base.PathLike, name: str) -> Path:
    """Check that the parent path exists and create directory"""
    path = Path(path)
    if not path.parent.exists():
        raise RuntimeError(f"Parent folder {path.parent} of {name} must exist first.")
    path.mkdir(exist_ok=True)
    return path


def _identify_study_subfolder(path: str | Path, name: str) -> Path:
    """If provided with path and study name, if
    path / name or path / name.lower() exist then returns the first.
    This way path can be generic for all studies provided the actual study
    folder is a sub-folder with its name.
    """
    path = Path(path)
    path = _check_folder_path(path, name=f"{name}.path")
    if path.name.lower() != name.lower():
        # use the subfolder with capitalized or uncapitalized name if it exists,
        # this enables using same folder everywhere
        for n in (name, name.lower()):
            if (path / n).exists():
                logger.debug("Updating study path to %s", path)
                return path / n
    return path


def _set_dir_permissions(path: Path) -> None:
    """Recursively set 777 permissions on a directory (Unix only).

    Skips files/directories not owned by the current user, since only the
    owner (or root) may chmod.  This is expected on shared filesystems where
    another user originally downloaded the study.
    """
    if os.name == "nt":
        logger.info("Skipping permission setting on Windows.")
        return
    current_uid = os.getuid()
    skipped = 0
    for root, dirs, files in os.walk(path):
        for item in [root] + [os.path.join(root, n) for n in dirs + files]:
            try:
                if os.stat(item).st_uid != current_uid:
                    skipped += 1
                    continue
                os.chmod(item, 0o777)
            except PermissionError:
                logger.debug("Cannot chmod %s (not owner), skipping.", item)
                skipped += 1
    if skipped:
        logger.info("Skipped %d items in %s not owned by current user.", skipped, path)
    logger.info(f"Permissions set for {path} (skipped {skipped} non-owned items).")


def _scan_package_for_studies(base_dir: Path, base_module: str, name: str) -> None:
    """Scan a package directory tree for study modules and import those
    that may define a class matching *name*.

    When *name* is empty (called from :meth:`Study.catalog`), every non-test
    module is imported so that all ``__init_subclass__`` registrations fire.

    Matches class definitions (``class Name``), module-level aliases
    (``Name =``), and string literals (``"Name"``).
    """
    for fp in base_dir.rglob("*.py"):
        if fp.name.startswith(("test_", "_")) or "-" in fp.stem:
            continue
        rel = fp.relative_to(base_dir).with_suffix("")
        module = base_module + "." + ".".join(rel.parts)
        try:
            text = fp.read_text("utf8")
        except FileNotFoundError:
            continue  # editable installs can race with file creation
        if not name or name in text:
            importlib.import_module(module)
            if name and name in STUDIES:
                logger.debug("study %r found in %s", name, module)
                return


@functools.lru_cache(maxsize=1)
def _get_study_packages() -> tuple[str, ...]:
    """Return deduplicated module paths registered under the
    ``neuralset.studies`` entry-point group.  Each value is a dotted
    module path used as the scan root for study discovery.
    """
    packages = dict.fromkeys(
        ep.value for ep in importlib.metadata.entry_points(group="neuralset.studies")
    )
    result = tuple(packages)
    logger.debug("study packages discovered via entry points: %s", result)
    return result


def _resolve_study(name: str = "") -> tp.Type["Study"] | None:
    """Look up *name* in STUDIES, scanning external packages if needed.

    Returns the class if found, None otherwise.
    Pass empty string to trigger a full scan of all study packages.
    """
    if name and name in STUDIES:
        return STUDIES[name]
    for pkg_name in _get_study_packages():
        if name and name in STUDIES:
            break
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        if pkg.__file__ is None:
            continue
        _scan_package_for_studies(Path(pkg.__file__).parent, pkg_name, name)
    cls = STUDIES.get(name)
    if name and cls is None:
        scanned = ", ".join(_get_study_packages())
        hint = "import the module that defines it before use"
        if importlib.util.find_spec("neuralfetch") is None:
            hint = f"run `pip install neuralfetch` for public-data studies, or {hint}"
        raise ImportError(f"Study {name!r} not found (scanned: {scanned}), {hint}.")
    return cls


class StudyInfo(base.BaseModel):
    """Records expected dataset characteristics for testing and validation.

    Provides a baseline for automatic unit tests to verify that a study's
    data loading logic and file parsing behave correctly and completely.

    Attributes
    ----------
    num_timelines : int
        The total number of timelines (e.g. subject sessions) expected.
    num_subjects : int
        The expected number of unique subjects.
    query : str
        A query applied during tests to subsample the data (default: ``"timeline_index < 1"``).
    num_events_in_query : int
        The expected number of events after applying the query.
    event_types_in_query : set of str
        The expected set of event types present in the queried data.
    data_shape : tuple of int
        The expected shape of the primary data arrays.
    frequency : float
        The expected sampling frequency of the data, in Hz.
    fmri_spaces : tuple of str
        (fMRI only) The expected spatial reference spaces for the data.
    """

    model_config = pydantic.ConfigDict(extra="forbid")
    num_timelines: int = 0
    num_subjects: int = 0
    # from query
    query: str = "timeline_index < 1"
    num_events_in_query: int = 0
    event_types_in_query: set[str] = set()
    data_shape: tuple[int, ...] = ()
    frequency: float = 0.0
    # for FMRI only
    fmri_spaces: tuple[str, ...] | set[str] = ()


# Lives here (not in transforms/) to avoid circular import: events → transforms → extractors
class EventsTransform(base.Step):
    """Base class for steps that modify an :term:`events <event>` dataframe.

    Transforms take an input events DataFrame, modify it (e.g. by adding new
    columns, filtering rows, or deriving new events), and return the modified
    DataFrame.

    Subclasses should override ``_run(self, events: pd.DataFrame) -> pd.DataFrame``.

    Examples
    --------
    .. code-block:: python

        class MyFilter(ns.EventsTransform):
            def _run(self, events: pd.DataFrame) -> pd.DataFrame:
                return events[events["type"] == "Audio"]
    """

    _DEFAULT_CACHE_TYPE: tp.ClassVar[str | None] = "ValidatedParquet"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "_run" in cls.__dict__:
            original = cls.__dict__["_run"]

            @functools.wraps(original)
            def _validated_run(self: tp.Any, *args: tp.Any, **kw: tp.Any) -> pd.DataFrame:
                result = original(self, *args, **kw)
                return utils.standardize_events(result, auto_fill=False)

            cls._run = _validated_run  # type: ignore[assignment]

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        """Standalone usage: delegates directly to _run."""
        return self._run(events)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# STUDY V2
STUDY_PATHS: dict[str, Path] = {}
STUDIES: dict[str, tp.Type["Study"]] = {}


def _dict_to_kv(d: dict[str, tp.Any]) -> str:
    """Format dict as ``key=val,key2=val2`` with sorted keys."""
    return ",".join(f"{k}={v}" for k, v in sorted(d.items()))


class Study(base.Step):
    """Interface to an external dataset: loads :term:`events <event>` from raw recordings.

    Subclass ``Study`` to create an interface to a new dataset. Override
    :meth:`iter_timelines` to enumerate :term:`timelines <timeline>` and
    :meth:`_load_timeline_events` to load events for each one.

    Parameters
    ----------
    path : Path
        Root directory for the study data. All studies can use the same
        ``path`` (e.g. ``path="/data"``): each study automatically
        resolves its own subfolder (e.g. ``/data/MyStudy``), so you
        only need to configure one path for your entire data store.
    infra_timelines : MapInfra
        Caching/compute backend for per-timeline event loading. Uses
        multiprocessing by default (``cluster="processpool"``); set
        ``cluster=None`` to disable (slower, but easier to debug).
    query : Query or None
        Optional filter applied after loading (e.g. ``"timeline_index < 5"``).

    Examples
    --------
    .. code-block:: python

        # Direct instantiation:
        study = MyStudy(path="/data/studies")
        # path auto-resolves to /data/studies/MyStudy if that subfolder exists
        events = study.run()

        # By name (auto-imports the study class):
        study = Study(name="MyStudy", path="/data/studies")
        events = study.run()

    .. note::

        Studies reference third-party datasets that are subject to their own
        licenses. You may have other legal obligations or restrictions that
        govern your use of that content. Check each study's :attr:`licence`
        and :attr:`url` attributes for details.

    """

    _DEFAULT_CACHE_TYPE: tp.ClassVar[str | None] = "ValidatedParquet"

    path: Path
    infra_timelines: exca.MapInfra = exca.MapInfra(cluster="processpool")
    query: base.Query | None = None

    # internal
    _cls_string: str = ""  # cache

    # Class level info
    _info: tp.ClassVar[None | StudyInfo] = None  # for easy testing
    aliases: tp.ClassVar[tuple[str, ...]] = ()
    url: tp.ClassVar[str] = ""
    bibtex: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = ""
    description: tp.ClassVar[str] = ""

    @classmethod
    def catalog(cls) -> dict[str, tp.Type["Study"]]:
        """All registered Study subclasses, keyed by name.

        Triggers lazy imports so that studies from all installed
        packages (e.g. neuralfetch) are discovered.

        Returns
        -------
        dict[str, type[Study]]
            ``{name: StudySubclass}`` for every registered study.
        """
        _resolve_study()
        out = dict(STUDIES)
        if not out:
            raise ImportError(
                "No studies found. Install a study package (e.g. pip install neuralfetch) "
                "or import your own Study subclass before calling catalog()."
            )
        return out

    @classmethod
    def neuro_types(cls) -> frozenset[str]:
        """Neural recording event types from ``_info``, empty if ``_info`` is not set."""
        if cls._info is None:
            return frozenset()
        names = set(etypes.EventTypesHelper(("MneRaw", "Fmri")).names)
        return frozenset(cls._info.event_types_in_query & names)

    if tp.TYPE_CHECKING:

        def __init__(  # pylint: disable=super-init-not-called
            self, **kwargs: tp.Any
        ) -> None: ...  # override needed: __new__ breaks mypy field detection

    def __new__(cls, /, **kwargs: tp.Any) -> "Study":
        # __new__ because discovery must happen before DiscriminatedModel dispatches
        # (model_validator runs too late — after __new__ already picked the class)
        name = kwargs.get(cls._exca_discriminator_key, "")
        if name:
            _resolve_study(name)
        return super().__new__(cls, **kwargs)  # type: ignore[return-value]

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        raise NotImplementedError
        # # example:
        # for subject in range(20):
        #     for session in range(10):
        #         yield dict(subject=subject, session=session)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raise NotImplementedError
        # # example
        # event = dict(
        #     type="Eeg",
        #     subject=params["subject"],
        #     start=0,
        #     duration=100,
        #     filepath=f"{params['subject']}-{params['session']}.fif",
        # )
        # return pd.DataFrame([event])

    def _download(self) -> None:
        """Download dataset.
        Needs to be overriden by user.
        """
        raise NotImplementedError("Dataset not available to download yet.")

    @tp.final
    def download(self, **kwargs: tp.Any) -> None:
        self._check_requirements()
        # Ensure download goes into a subfolder named after the study
        name = self.__class__.__name__
        if self.path.name.lower() != name.lower():
            self.path = self.path / name
            STUDY_PATHS[self.__class__.__name__] = self.path
            logger.info("Download path updated to %s", self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._download(**kwargs)
        if not self.path.exists():
            raise RuntimeError(f"Path does not exist: {self.path}")
        if not self.path.is_dir():
            raise RuntimeError(f"Path is not a directory: {self.path}")
        if not any(self.path.iterdir()):
            raise RuntimeError(f"Directory is empty: {self.path}")
        logger.info(f"Success: Study downloaded to {self.path}.")
        _set_dir_permissions(self.path)
        self.clear_cache()

    def _cls_kwargs(self) -> dict[str, tp.Any]:
        """Descriptor for the study instance parametrization"""
        cls_kwargs: tp.Any = self.model_dump(serialize_as_any=True, exclude_defaults=True)
        # Exclude standard fields from class kwargs
        for p in ["infra", "infra_timelines", "path", "name", "query"]:
            cls_kwargs.pop(p, None)
        if cls_kwargs:
            # should the class parameter be part of the timeline? or does
            # it select a subset? the behavior is unclear and should be
            # specified precisely first.
            msg = "Class parameters are not yet supported, bring up your use-case!"
            raise RuntimeError(msg)

        return cls_kwargs

    def _to_timeline_string(self, timeline: dict[str, tp.Any]):
        if not self._cls_string:
            self._cls_string = self.__class__.__name__
            cls_kwargs = self._cls_kwargs()
            if cls_kwargs:
                self._cls_string += ":" + _dict_to_kv(cls_kwargs)
        return self._cls_string + ":" + _dict_to_kv(timeline)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["path"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if type(self) is Study:
            raise TypeError(
                "Study cannot be instantiated directly — use a subclass, "
                "or pass name= to dispatch: Study(name='MyStudy2024', path=...)"
            )
        name = self.__class__.__name__
        self.path = _identify_study_subfolder(self.path, name)
        STUDY_PATHS[self.__class__.__name__] = self.path  # record for path lookup
        # Auto-propagate cache folder and mode so users only need to set it once
        if self.infra is not None:
            if self.infra.folder is not None and self.infra_timelines.folder is None:
                self.infra_timelines.folder = self.infra.folder
            if "mode" in self.infra.model_fields_set:
                if "mode" not in self.infra_timelines.model_fields_set:
                    mode = self.infra.mode
                    self.infra_timelines.mode = "cached" if mode == "retry" else mode

    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        name = cls.__name__
        super().__init_subclass__(**kwargs)
        if not name.startswith("_"):
            existing = STUDIES.get(name)
            if existing is not None and existing is not cls:
                raise RuntimeError(
                    f"Study name collision: {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}, "
                    f"cannot re-register to {cls.__module__}.{cls.__qualname__}"
                )
            STUDIES[name] = cls
        if hasattr(cls, "version"):
            msg = (
                f"{name}.version must be specified through {name}.infra_timelines.version"
            )
            raise RuntimeError(msg)

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        super().__setstate__(state)
        # reregister study path
        STUDY_PATHS[self.__class__.__name__] = self.path

    @infra_timelines.apply(
        item_uid=lambda x: ujson.dumps(x, sort_keys=True),
        exclude_from_cache_uid=("query",),
        cache_type="ValidatedParquet",  # preserves str dtypes (CSV doesn't)
    )
    def _load_timelines(
        self, timelines: tp.Iterable[dict[str, tp.Any]]
    ) -> tp.Iterator[pd.DataFrame]:
        """Loads raw timelines and cache them"""
        cls_name = self.__class__.__name__
        for timeline in timelines:
            if "subject" not in timeline:
                raise RuntimeError("timeline dict must contain 'subject' key")
            out = self._load_timeline_events(timeline)
            # Core columns are always overwritten
            out.loc[:, "subject"] = f"{cls_name}/{timeline['subject']}"
            out.loc[:, "timeline"] = self._to_timeline_string(timeline)
            out.loc[:, "study"] = cls_name
            # Extra columns from timeline dict: conflict-check then set
            extra: dict[str, str] = {}
            for key, value in timeline.items():
                if key not in ("subject", "path", "timeline"):
                    extra[key] = str(value)
            for entity in utils.BIDS_ENTITIES:
                if entity != "subject":
                    extra.setdefault(entity, utils.BIDS_ENTITY_DEFAULT)
            for col, value in extra.items():
                if col not in out.columns:
                    out.loc[:, col] = value
                elif (out[col].astype(str) == value).all():
                    warnings.warn(
                        f"Column '{col}' from timeline dict already exists "
                        f"in the events dataframe with matching values. "
                        f"Remove it from _load_timeline_events to avoid "
                        f"future errors.",
                        FutureWarning,
                    )
                elif (
                    col in utils.BIDS_ENTITIES
                    and (out[col].astype(str) == utils.BIDS_ENTITY_DEFAULT).all()
                ):
                    out.loc[:, col] = value
                else:
                    raise ValueError(
                        f"Column '{col}' from timeline dict already exists "
                        f"in the events dataframe with different values."
                    )
            out = utils.standardize_events(out)
            yield out

    def _run(self, events: pd.DataFrame | None = None) -> pd.DataFrame:
        """Load study data and optionally concatenate with existing events."""
        if events is not None:  # special case, concatenate
            df = self._run()
            return pd.concat([events, df], ignore_index=True).reset_index(drop=True)
        name = self.__class__.__name__
        timelines = []
        # iterate 1 by 1 to provide explicit msg to different kinds of bugs
        try:
            for tl in self.iter_timelines():
                timelines.append(tl)
        except Exception as e:
            # for a bug on the first timeline with no folder, raise with more info,
            if not timelines and not self.path.exists():
                msg = f"For {name}, you may need to run study.download() first "
                msg += f"as {self.path} does not exist."
                raise RuntimeError(msg) from e
            raise
        if not timelines:
            raise RuntimeError(f"No timeline found for {name} in {self.path}")
        # verify number of timelines in info
        if self._info is not None:
            tls = self._info.num_timelines
            if tls != len(timelines):
                msg = f"Dataset {name} is corrupted, expected {tls} timelines "
                msg += f"but found {len(timelines)} (check/redownload dataset "
                msg += f"folder {self.path} or update study class)"
                raise RuntimeError(msg)
        # filter through summary
        if self.query is not None:
            summ = self.study_summary(apply_query=True)
            timelines = [timelines[k] for k in summ.index]
        if not timelines:
            msg = f"Did not find any timeline for {name}.query={self.query} "
            msg += f"with summary:\n{self.study_summary(apply_query=False)}"
            raise RuntimeError(msg)
        # load and concatenate
        timelines_events = list(self._load_timelines(timelines))
        out = pd.concat(timelines_events).reset_index(drop=True)
        return utils.standardize_events(out, auto_fill=False)

    def build(self) -> pd.DataFrame:
        """alias for run"""
        return self.run()

    def study_summary(self, apply_query: bool = True) -> pd.DataFrame:
        """Returns a dataframe with 1 row per timeline and study attributes as columns.
        :code:`query` parameter is used on this dataframe for subselection

        Parameter
        ---------
        apply_query: bool
            if False returns the full the summary, otherwise filter it
            according to the query

        Virtual query columns
        ---------------------
        The following columns are **not** present in the returned DataFrame
        but are auto-generated by :func:`~neuralset.events.transforms.query_with_index`
        when ``apply_query=True`` and the ``query`` string references them:

        :code:`subject_index`: int
            the index of the subject in the study
        :code:`timeline_index`: int
            the index of the timeline in the study (equivalent to "index")
        :code:`subject_timeline_index`: int
            the index of the timeline among a subject's timelines in the study
            (used for querying at most :code:`n` timelines per subjects)
        """
        name = self.__class__.__name__
        tls = list(self.iter_timelines())
        out = pd.DataFrame(tls)
        if out.empty:
            raise RuntimeError(f"No timeline found for {self!r}")
        out["subject"] = out["subject"].apply(lambda x: f"{name}/{x}")
        out.loc[:, "timeline"] = [self._to_timeline_string(tl) for tl in tls]
        out = out.sort_values("subject", kind="stable")
        if apply_query and self.query is not None:
            out = utils.query_with_index(out, self.query)
        return out
