# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Datasets from the Mother of all BCI Benchmarks (MoABB), a collection of EEG datasets for BCI.


References
----------
MoABB software
bibtex = @software{Aristimunha_Mother_of_all,
    author = {
        Aristimunha, Bruno and
        Carrara, Igor and
        Guetschel, Pierre and
        Sedlar, Sara and
        Rodrigues, Pedro and
        Sosulski, Jan and
        Narayanan, Divyesh and
        Bjareholt, Erik and
        Barthelemy, Quentin and
        Schirrmeister, Robin Tibor and
        Kobler, Reinmar and
        Kalunga, Emmanuel and
        Darmet, Ludovic and
        Gregoire, Cattan and
        Abdul Hussain, Ali and
        Gatti, Ramiro and
        Goncharenko, Vladislav and
        Thielen, Jordy and
        Moreau, Thomas and
        Roy, Yannick and
        Jayaram, Vinay and
        Barachant, Alexandre and
        Chevallier, Sylvain},
    title        = {Mother of all BCI Benchmarks},
    year         = 2025,
    publisher    = {Zenodo},
    version      = {v1.2.0},
    url = {https://github.com/NeuroTechX/moabb},
    doi = {10.5281/zenodo.10034223},
}
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import typing as tp
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from neuralfetch.download import temp_mne_data
from neuralset.events import study as studies
from neuralset.events.study import STUDY_PATHS

if tp.TYPE_CHECKING:
    from moabb.datasets.base import BaseDataset as MoabbBaseDataset

logger = logging.getLogger(__name__)
logger.propagate = False


def _patch_figshare_file_list_cache() -> None:
    """Cache ``moabb.datasets.download.fs_get_file_list`` to avoid repeated
    Figshare API calls when data is already on disk.  MOABB's MAMEM loaders
    call the API on *every* ``get_data`` invocation, which triggers 403 rate
    limits from datacenter IPs."""
    try:
        from moabb.datasets import (  # type: ignore[import-untyped,import-not-found]
            download as _dl,
        )

        if not getattr(_dl.fs_get_file_list, "_patched", False):
            _orig = _dl.fs_get_file_list

            @lru_cache(maxsize=32)
            def _cached_fs_get_file_list(article_id, version=None):  # type: ignore[no-untyped-def]
                return _orig(article_id, version=version)

            _cached_fs_get_file_list._patched = True  # type: ignore[attr-defined]
            _dl.fs_get_file_list = _cached_fs_get_file_list
            # Also patch in ssvep_mamem where it's imported directly
            try:
                from moabb.datasets import ssvep_mamem as _mamem

                _mamem.fs_get_file_list = _cached_fs_get_file_list
            except (ImportError, AttributeError):
                pass
    except ImportError:
        pass


_patch_figshare_file_list_cache()


# Module-level bounded cache used by classes that bypass _get_cached_moabb_data
# (e.g. Romani2025Brainform caches raw objects directly).
_moabb_data_cache: OrderedDict[tuple[str, ...], tp.Any] = OrderedDict()
_MOABB_CACHE_MAXSIZE = 2


@lru_cache(maxsize=2)
def _get_cached_moabb_data(
    dataset_name: str,
    dl_path: Path,
    subject: int,
    dataset_kwargs: tuple[tuple[str, tp.Any], ...] | None = None,
) -> dict[tp.Any, tp.Any]:
    """Load a single subject via MOABB with module-level LRU caching."""
    from moabb.datasets.base import CacheConfig

    kwargs_dict = dict(dataset_kwargs) if dataset_kwargs else None
    with temp_mne_data(dl_path, clear_dataset_configs=True):
        ds = find_dataset_in_moabb(dataset_name, kwargs_dict)
        data = ds.get_data(subjects=[subject], cache_config=CacheConfig(path=dl_path))
    return data


def find_dataset_in_moabb(
    dataset_name: str,
    dataset_kwargs: dict[str, tp.Any] | None = None,
) -> type[MoabbBaseDataset]:
    """Find and instantiate a MOABB dataset by name."""
    from moabb.datasets.utils import dataset_list

    accept = os.environ.get("MOABB_ACCEPT_LICENCE", "").lower() in ("1", "true", "yes")
    for dataset_cls in dataset_list:
        if dataset_cls.__name__ == dataset_name:
            kwargs = dict(dataset_kwargs or {})
            init_params = inspect.signature(dataset_cls.__init__).parameters
            if "accept" in init_params and "accept" not in kwargs:
                if not accept:
                    raise RuntimeError(
                        f"Dataset '{dataset_name}' requires accepting a licence "
                        f"agreement before downloading. Set the environment "
                        f"variable MOABB_ACCEPT_LICENCE=1 to accept."
                    )
                kwargs["accept"] = True
            return dataset_cls(**kwargs)

    raise ValueError(f"MOABB dataset not found: {dataset_name}")


class _BaseMoabb(studies.Study):
    _dataset_kwargs: tp.ClassVar[dict[str, tp.Any]] = {}
    event_id: tp.ClassVar[dict[str, tp.Any]] = {}
    requirements: tp.ClassVar[tuple[str, ...]] = ("moabb>=1.5.0",)

    def _rename_numeric_descriptions(self, events_df: pd.DataFrame) -> None:
        """Auto-rename numeric descriptions inferred from event_id keys.

        If all keys are numeric and integer-valued (e.g. "0", "1.0"),
        descriptions are prefixed with ``code_``.  If at least one key
        has a fractional float value (e.g. "8.57"), descriptions are
        suffixed with ``Hz`` (stimulation frequency).
        """
        keys = list(self.event_id.keys())
        if not keys:
            return
        try:
            vals = [float(k) for k in keys]
        except (ValueError, TypeError):
            return
        if all(v == int(v) for v in vals):
            events_df["description"] = events_df["description"].map(
                lambda x: f"code_{int(float(x))}"
            )
        else:
            events_df["description"] = events_df["description"].map(lambda x: f"{x}Hz")

    def model_post_init(  # pylint: disable=attribute-defined-outside-init
        self, log__: tp.Any
    ) -> None:
        """Override path handling to group all MOABB studies under 'moabb' subfolder.

        Transforms: /path/StudyName -> /path/moabb/StudyName
        """
        # First, let the parent class handle basic path setup
        # It will transform /path -> /path/StudyName
        super().model_post_init(log__)

        # Disable the processpool for MOABB datasets: some (e.g. ERP CORE)
        # load via mne_bids which uses file locks that exhaust NFS NLM limits
        # when too many workers run in parallel (ENOLCK).  Timeline events are
        # cached on disk by exca, so sequential loading only costs on first run.
        if self.infra_timelines.cluster == "processpool":
            self.infra_timelines = self.infra_timelines.model_copy(
                update={"cluster": None}
            )

        # Now insert 'moabb' before the study name
        # self.path is now /path/StudyName, we want /path/moabb/StudyName
        study_name = self.__class__.__name__

        # Check if 'moabb' is already in the path (e.g., /path/moabb/StudyName)
        if "moabb" not in self.path.parts:
            # Strip the study-name suffix if the parent class appended it
            # (case-insensitive: _identify_study_subfolder may return lowercase)
            if self.path.name.lower() == study_name.lower():
                parent = self.path.parent
            else:
                parent = self.path
            self.path = parent / "moabb" / study_name

            self.path.mkdir(parents=True, exist_ok=True)
            # Update the STUDY_PATHS registry with the corrected path
            STUDY_PATHS[self.__class__.__name__] = self.path
            logger.debug("Updated MOABB path to: %s", self.path)

    @classmethod
    def _get_dataset(cls) -> type[MoabbBaseDataset]:
        return find_dataset_in_moabb(cls.aliases[0], cls._dataset_kwargs or None)

    def _timelines_df(self) -> pd.DataFrame:
        df = pd.read_csv(Path(self.path) / "timelines.csv", header=0)
        df["subject"] = df["subject"].astype(int)
        df["session"] = df["session"].astype(str)
        df["run"] = df["run"].astype(str)
        return df

    # Populate subclass-level metadata
    def _download(self) -> None:
        """
        Download MoABB dataset and write timelines as a csv.

        (Timelines needed to avoid 'moabb.datasets.studies.get_data()' in self.iter_timelines, which loads ALL raw data.)

        """
        from moabb.datasets.base import CacheConfig

        timeline_path = Path(self.path) / "timelines.csv"

        if not timeline_path.exists():
            dl_path = Path(self.path) / "download"
            try:
                dl_path.mkdir(exist_ok=True, parents=True)
                logger.info(f"Created download directory: {dl_path}")
                with temp_mne_data(dl_path, clear_dataset_configs=True):
                    logger.info(f"MNE_DATA set to: {dl_path}")
                    ds = self._get_dataset()
                    rows: list[dict[str, tp.Any]] = []
                    failed_subjects: list[tp.Any] = []
                    for subject in ds.subject_list:
                        try:
                            subj_data = ds.get_data(
                                subjects=[subject],
                                cache_config=CacheConfig(path=dl_path),
                            )
                        except Exception as e:
                            logger.error(
                                f"Error reading subject {subject} of "
                                f"{self.aliases[0]}: {e}"
                            )
                            failed_subjects.append(subject)
                            continue
                        for session, runs in subj_data[subject].items():
                            for run in runs:
                                rows.append(
                                    dict(
                                        subject=int(subject),
                                        session=str(session),
                                        run=str(run),
                                    )
                                )
                    if failed_subjects:
                        logger.warning(
                            f"Failed {len(failed_subjects)} subject(s) for "
                            f"{self.aliases[0]}: {failed_subjects}"
                        )

                timelines = pd.DataFrame(rows)
                assert not timelines.empty, (
                    f"Failed to create timelines.csv at {timeline_path}"
                )
                timelines.to_csv(timeline_path, index=False)
                logger.info(
                    f"Downloaded dataset to {dl_path} "
                    f"({len(timelines)} timelines, "
                    f"{len(failed_subjects)} failed subjects)."
                )
            except Exception as error:
                logger.error(f"Download failed: {error}")
                raise

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings."""
        for row in self._timelines_df().itertuples():
            yield {
                "subject": row.subject,
                "session": row.session,
                "run": row.run,
            }

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        info = studies.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        # start="auto": auto-resolve from raw.first_samp
        eeg = dict(type="Eeg", start="auto", filepath=info)
        eeg_df = pd.DataFrame([eeg])
        # Events from MNE Annotations.
        # Project to known scalar columns: ``to_data_frame`` would otherwise
        # flatten ``Annotations.extras`` into extra columns whose values may
        # be dicts, which ``ValidatedParquet`` cannot serialize.
        events_df = raw.annotations.to_data_frame(time_format=None)
        events_df = events_df[["onset", "duration", "description"]]
        events_df = events_df.rename(columns=dict(onset="start"))
        events_df["description"] = events_df.description.astype(str)
        has_list_codes = any(isinstance(v, list) for v in self.event_id.values())
        if has_list_codes:
            # Multi-code event_id (e.g. {"Target": [111, 112, ...]}):
            # assign a category index as code, keep original description as-is,
            # and store the raw MOABB codes in a 'stimulus' column (extra).
            category_map = {label: i for i, label in enumerate(self.event_id)}
            events_df["code"] = events_df.description.map(category_map)
            events_df["stimulus"] = events_df.description.map(
                {label: str(codes) for label, codes in self.event_id.items()}
            )
        else:
            events_df["code"] = events_df.description.map(self.event_id)
        self._rename_numeric_descriptions(events_df)
        events_df.insert(0, "type", "Stimulus")
        result = pd.concat([eeg_df, events_df]).reset_index(drop=True)
        if "description" in result.columns:
            result["description"] = result["description"].fillna("").astype(str)
        return result

    @staticmethod
    def _dict_get(d: dict, key: str) -> tp.Any:
        """Look up *key* in *d*, tolerating int/str mismatches.

        MOABB ``get_data()`` uses int session/run keys for some datasets
        but ``timelines.csv`` stores them as strings.
        """
        try:
            return d[key]
        except KeyError:
            pass
        try:
            return d[int(key)]
        except (KeyError, ValueError):
            raise KeyError(key) from None

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        tl = timeline
        dl_path = Path(self.path) / "download"

        if not dl_path.exists():
            raise FileNotFoundError(
                f"Data directory not found at {dl_path}. "
                f"Please run download() first to download the dataset."
            )

        subject = tl["subject"]
        kw = tuple(sorted(self._dataset_kwargs.items())) if self._dataset_kwargs else None
        data = _get_cached_moabb_data(self.aliases[0], dl_path, subject, kw)
        sessions = data[subject]
        runs = self._dict_get(sessions, tl["session"])
        return self._dict_get(runs, tl["run"])


# ============================================================================
# DATASET CLASSES
# ============================================================================


class _BaseCvepTrialMoabb(_BaseMoabb):
    """c-VEP study base: extracts trial-level (symbol) events from stim_trial.

    Subclasses may set ``_code_length`` and ``_presentation_rate`` to enable
    automatic cycle-level splitting.  When both are set, each trial produces
    one Stimulus event per code cycle whose onset is computed as::

        trial_onset + cycle_index * _code_length / _presentation_rate

    Datasets whose ``_code_length`` varies across sessions / runs (e.g.
    ``MartinezCagigal2023Pary`` with 5 different m-sequence bases) should
    override :meth:`_get_code_length` instead of setting the class var.
    """

    _trial_stim_offset: tp.ClassVar[int] = 200
    _code_length: tp.ClassVar[int | None] = None
    _presentation_rate: tp.ClassVar[float | None] = None
    _n_cycles: tp.ClassVar[int | None] = None
    _trial_duration: tp.ClassVar[float | None] = None

    def _get_code_length(self, timeline: dict[str, tp.Any]) -> int | None:
        """Return the m-sequence length (in bits) for ``timeline``.

        Default returns the class-level :attr:`_code_length`. Override when the
        code length depends on the timeline (e.g. per-session m-sequence base).
        """
        del timeline
        return self._code_length

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        info = studies.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame([dict(type="Eeg", start="auto", filepath=info)])

        trial_events = mne.find_events(
            raw, stim_channel="stim_trial", shortest_event=0, verbose=False
        )
        sfreq = raw.info["sfreq"]
        code_length = self._get_code_length(timeline)

        rows: list[dict[str, tp.Any]] = []
        for onset_samp, _, marker in trial_events:
            symbol_id = int(marker - self._trial_stim_offset)
            trial_onset = onset_samp / sfreq

            if code_length is not None and self._presentation_rate is not None:
                cycle_dur = code_length / self._presentation_rate
                n_cycles = self._n_cycles_in_trial(
                    onset_samp, trial_events, raw.n_times, sfreq, cycle_dur
                )
                if self._n_cycles is not None:
                    n_cycles = min(n_cycles, self._n_cycles)
                for c in range(n_cycles):
                    rows.append(
                        {
                            "start": trial_onset + c * cycle_dur,
                            "duration": cycle_dur,
                            "description": f"symbol_{symbol_id}",
                            "code": symbol_id,
                        }
                    )
            else:
                assert self._trial_duration is not None, (
                    f"{type(self).__name__} must set _trial_duration or "
                    f"_code_length/_presentation_rate"
                )
                rows.append(
                    {
                        "start": trial_onset,
                        "duration": self._trial_duration,
                        "description": f"symbol_{symbol_id}",
                        "code": symbol_id,
                    }
                )

        events_df = pd.DataFrame(rows)
        events_df.insert(0, "type", "Stimulus")
        result = pd.concat([eeg_df, events_df]).reset_index(drop=True)
        result["description"] = result["description"].fillna("").astype(str)
        return result

    @staticmethod
    def _n_cycles_in_trial(
        onset_samp: int,
        all_trial_events: np.ndarray,
        n_times: int,
        sfreq: float,
        cycle_dur: float,
    ) -> int:
        """Return the number of full cycles that fit in the current trial."""
        idx = int(np.searchsorted(all_trial_events[:, 0], onset_samp))
        if idx + 1 < len(all_trial_events):
            next_onset = all_trial_events[idx + 1, 0] / sfreq
        else:
            next_onset = n_times / sfreq
        available = next_onset - onset_samp / sfreq
        return int(available // cycle_dur)


class Acqualagna2013Gaze(_BaseMoabb):
    """Subset of MOABB: BNCI2015_010"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_010",)
    bibtex: tp.ClassVar[str] = """
        @article{acqualagna2013gaze,
          title={Gaze-independent BCI-spelling using rapid serial visual presentation (RSVP)},
          volume={124},
          ISSN={1388-2457},
          url={http://dx.doi.org/10.1016/j.clinph.2012.12.050},
          DOI={10.1016/j.clinph.2012.12.050},
          number={5},
          journal={Clinical Neurophysiology},
          publisher={Elsevier BV},
          author={Acqualagna, Laura and Blankertz, Benjamin},
          year={2013},
          month=may,
          pages={901-908}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (63-ch) in 12 healthy participants while attending to target letters in a rapid serial visual presentation (RSVP) P300 speller."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=24,
        num_subjects=12,
        num_events_in_query=7201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 492552),
        frequency=200.0,
    )


class Arico2014Influence(_BaseMoabb):
    """Subset of MOABB: BNCI2014_009"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2014_009",)
    bibtex: tp.ClassVar[str] = """
        @article{arico2014influence,
          title={Influence of P300 latency jitter on event related potential-based brain--computer interface performance},
          author={Aricò, Pietro and Aloise, Fabio and Schettini, Francesca and Salinari, Serenella and Mattia, Donatella and Cincotti, Febo},
          journal={Journal of Neural Engineering},
          volume={11},
          number={3},
          pages={035008},
          year={2014},
          publisher={IOP Publishing},
          doi={10.1088/1741-2560/11/3/035008}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 10 healthy participants while performing a P300 speller task using row-column (6x6 matrix) and GeoSpell (hexagonal) paradigms."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=30,
        num_subjects=10,
        num_events_in_query=577,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 50184),
        frequency=256.0,
    )


class Barachant2012Robust(_BaseMoabb):
    """Subset of MOABB: AlexMI"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("AlexMI",)
    bibtex: tp.ClassVar[str] = """
        @phdthesis{barachant2012robust,
          TITLE = {Robust control of an effector by an asynchronous EEG brain-machine interface},
          AUTHOR = {Barachant, Alexandre},
          URL = {https://theses.hal.science/tel-01196752},
          NUMBER = {2012GRENT112},
          SCHOOL = {{Universit{e} de Grenoble}},
          YEAR = {2012},
          MONTH = Mar,
          TYPE = {Theses},
          PDF = {https://theses.hal.science/tel-01196752v1/file/BARACHANT_2012_archivage.pdf},
          HAL_ID = {tel-01196752},
          HAL_VERSION = {v1},
          DOI = {10.5281/zenodo.806022}
        }

    @misc{barachant2012eeg,
          author={Alexandre Barachant},
          title={EEG Motor Imagery Dataset from the PhD Thesis "Commande robuste d'un effecteur par une interface cerveau machine EEG asynchrone"},
          month=mar,
          year={2012},
          publisher={Zenodo},
          doi={10.5281/zenodo.806023},
          url={https://doi.org/10.5281/zenodo.806023}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/806023/"
    licence: tp.ClassVar[str] = "CC-BY-SA-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (22-ch) in 8 participants while performing left-hand, right-hand, and feet motor imagery for asynchronous BCI control."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"right_hand": 2, "feet": 3, "rest": 4}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=8,
        num_subjects=8,
        num_events_in_query=61,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 254785),
        frequency=512.0,
    )


class CabreraCastillos2023BurstVep100(_BaseCvepTrialMoabb):
    """Subset of MOABB: CastillosBurstVEP100"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("CastillosBurstVEP100",)
    _trial_duration: tp.ClassVar[float] = 2.2
    bibtex: tp.ClassVar[str] = """
        @article{castillos2023burst,
          title={Burst c-VEP Based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience},
          volume={284},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2023.120446},
          DOI={10.1016/j.neuroimage.2023.120446},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Cabrera Castillos, Kalou and Ladouce, Simon and Darmet, Ludovic and Dehais, Frédéric},
          year={2023},
          month=dec,
          pages={120446}
        }

        @misc{cabreracastillos2023code,
          author={Cabrera Castillos, Kalou},
          title={4-class code-VEP EEG data},
          year={2023},
          publisher={Zenodo},
          doi={10.5281/zenodo.8255618},
          url={https://zenodo.org/records/8255618}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/8255618"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 12 healthy participants while performing a 4-class burst-VEP BCI task with 100 Hz visual stimulation."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(4)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=12,
        num_subjects=12,
        num_events_in_query=61,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 185508),
        frequency=500.0,
    )


class CabreraCastillos2023BurstVep40(_BaseCvepTrialMoabb):
    """Subset of MOABB: CastillosBurstVEP40"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("CastillosBurstVEP40",)
    _trial_duration: tp.ClassVar[float] = 2.2
    bibtex: tp.ClassVar[str] = """
        @article{castillos2023burst,
          title={Burst c-VEP Based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience},
          volume={284},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2023.120446},
          DOI={10.1016/j.neuroimage.2023.120446},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Cabrera Castillos, Kalou and Ladouce, Simon and Darmet, Ludovic and Dehais, Frédéric},
          year={2023},
          month=dec,
          pages={120446}
        }

        @misc{cabreracastillos2023code,
          author={Cabrera Castillos, Kalou},
          title={4-class code-VEP EEG data},
          year={2023},
          publisher={Zenodo},
          doi={10.5281/zenodo.8255618},
          url={https://zenodo.org/records/8255618}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/8255618"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 12 healthy participants while performing a 4-class burst-VEP BCI task with 40 Hz visual stimulation."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(4)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=12,
        num_subjects=12,
        num_events_in_query=61,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 132258),
        frequency=500.0,
    )


class CabreraCastillos2023BurstCvep100(_BaseCvepTrialMoabb):
    """Subset of MOABB: CastillosCVEP100"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("CastillosCVEP100",)
    _trial_duration: tp.ClassVar[float] = 2.2
    bibtex: tp.ClassVar[str] = """
        @article{castillos2023burst,
          title={Burst c-VEP Based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience},
          volume={284},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2023.120446},
          DOI={10.1016/j.neuroimage.2023.120446},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Cabrera Castillos, Kalou and Ladouce, Simon and Darmet, Ludovic and Dehais, Frédéric},
          year={2023},
          month=dec,
          pages={120446}
        }

        @misc{cabreracastillos2023code,
          author={Cabrera Castillos, Kalou},
          title={4-class code-VEP EEG data},
          year={2023},
          publisher={Zenodo},
          doi={10.5281/zenodo.8255618},
          url={https://zenodo.org/records/8255618}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/8255618"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 12 healthy participants while performing a 4-class code-modulated VEP (c-VEP) BCI task at 100 Hz."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(4)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=12,
        num_subjects=12,
        num_events_in_query=61,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 144517),
        frequency=500.0,
    )


class CabreraCastillos2023BurstCvep40(_BaseCvepTrialMoabb):
    """Subset of MOABB: CastillosCVEP40"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("CastillosCVEP40",)
    _trial_duration: tp.ClassVar[float] = 2.2
    bibtex: tp.ClassVar[str] = """
        @article{castillos2023burst,
          title={Burst c-VEP Based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience},
          volume={284},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2023.120446},
          DOI={10.1016/j.neuroimage.2023.120446},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Cabrera Castillos, Kalou and Ladouce, Simon and Darmet, Ludovic and Dehais, Frédéric},
          year={2023},
          month=dec,
          pages={120446}
        }

        @misc{cabreracastillos2023code,
          author={Cabrera Castillos, Kalou},
          title={4-class code-VEP EEG data},
          year={2023},
          publisher={Zenodo},
          doi={10.5281/zenodo.8255618},
          url={https://zenodo.org/records/8255618}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/8255618"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 12 healthy participants while performing a 4-class code-modulated VEP (c-VEP) BCI task at 40 Hz."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(4)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=12,
        num_subjects=12,
        num_events_in_query=61,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 128009),
        frequency=500.0,
    )


class Cattan2019Passive(_BaseMoabb):
    """Subset of MOABB: Cattan2019_PHMD"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Cattan2019_PHMD",)
    bibtex: tp.ClassVar[str] = """
        @misc{cattan2019passive,
          doi = {10.5281/ZENODO.2617084},
          url = {https://zenodo.org/record/2617084},
          author = {Cattan, Grégoire and Rodrigues, Pedro L. C. and Congedo, Marco},
          keywords = {Electroencephalography (EEG), P300, Brain-Computer Interface (BCI), Virtual Reality (VR), Experiment},
          title = {Passive Head-Mounted Display Music-Listening EEG dataset},
          publisher = {Zenodo},
          year = {2019},
          copyright = {Creative Commons Attribution 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2617085"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 12 healthy participants while passively listening to music wearing a head-mounted display at rest."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"on": 1, "off": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=12,
        num_subjects=12,
        num_events_in_query=11,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 412160),
        frequency=512.0,
    )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raw = super()._load_raw(timeline)
        # Guard against the lru_cache returning the same mutable RawArray:
        # only rename channels that haven't been renamed yet.
        rename = {
            k: v for k, v in {"Fc5": "FC5", "Fc6": "FC6"}.items() if k in raw.ch_names
        }
        if rename:
            raw.rename_channels(rename)
        raw.set_montage("standard_1020", on_missing="warn")
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Extend Stimulus durations to span from one marker to the next.

        MOABB annotations are 1-second markers, but each one marks the start of
        a ~75 s relaxation period. Extending durations enables stride windowing.
        """
        df = super()._load_timeline_events(timeline)
        stim_mask = df["type"] == "Stimulus"
        stim = df.loc[stim_mask].sort_values("start")
        starts = stim["start"]
        durations = starts.shift(-1) - starts
        eeg_row = df.loc[df["type"] == "Eeg"].iloc[0]
        if "duration" in eeg_row.index and pd.notna(eeg_row.get("duration")):
            rec_end = float(eeg_row["start"]) + float(eeg_row["duration"])
        else:
            rec_end = starts.iloc[-1] + 60.0
        durations.iloc[-1] = rec_end - starts.iloc[-1]
        df.loc[stim.index, "duration"] = durations.values
        return df


class Cattan2019Dataset(_BaseMoabb):
    """Subset of MOABB: Cattan2019_VR"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Cattan2019_VR",)
    bibtex: tp.ClassVar[str] = """
        @misc{cattan2019dataset,
          doi = {10.5281/ZENODO.2605204},
          url = {https://zenodo.org/record/2605204},
          author = {Cattan, Grégoire and Andreev, Anton and Rodrigues, Pedro L. C. and Congedo, Marco},
          keywords = {Electroencephalography (EEG), P300, Brain-Computer Interface (BCI), Virtual Reality (VR), Experiment},
          title = {Dataset of an EEG-based BCI experiment in Virtual Reality and on a Personal Computer},
          publisher = {Zenodo},
          year = {2019},
          copyright = {Creative Commons Attribution 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 21 healthy participants while performing a P300 BCI spelling task in virtual reality with a 6x6 flashing symbol matrix."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=1260,
        num_subjects=21,
        num_events_in_query=13,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 3261),
        frequency=512.0,
    )


class Chavarriaga2010Learning(_BaseMoabb):
    """Subset of MOABB: BNCI2015_013"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_013",)
    bibtex: tp.ClassVar[str] = """
        @article{chavarriaga2010learning,
          title={Learning From EEG Error-Related Potentials in Noninvasive Brain-Computer Interfaces},
          volume={18},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/TNSRE.2010.2053387},
          DOI={10.1109/tnsre.2010.2053387},
          number={4},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Chavarriaga, Ricardo and Millan, José del R.},
          year={2010},
          month=aug,
          pages={381-388}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 6 healthy participants while monitoring an autonomous cursor for error-related potentials (ErrP) in a navigation task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=120,
        num_subjects=6,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 91648),
        frequency=512.0,
    )


class Cho2017Supporting(_BaseMoabb):
    """Subset of MOABB: Cho2017"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Cho2017",)
    bibtex: tp.ClassVar[str] = """
        @misc{cho2017supporting,
          doi = {10.5524/100295},
          url = {http://gigadb.org/dataset/100295},
          author = {Cho, Hohyun and Ahn, Minkyu and Ahn, Sangtae and Kwon, Moonyoung and Jun, Sung, Chan},
          keywords = {ElectroEncephaloGraphy(EEG), motor imagery, eeg, brain computer interface, performance variation, subject-to-subject transfer},
          language = {en},
          title = {Supporting data for "EEG datasets for motor imagery brain computer interface"},
          publisher = {GigaScience Database},
          year = {2017},
          copyright = {Creative Commons Zero v1.0 Universal}
        }
    """
    url: tp.ClassVar[str] = (
        "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/"
    )
    licence: tp.ClassVar[str] = "UNKNOWN"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 52 healthy participants while performing left- and right-hand motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=52,
        num_subjects=52,
        num_events_in_query=201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 717300),
        frequency=512.0,
    )


class Vaineau2018Brain(_BaseMoabb):
    """Subset of MOABB: BI2013a"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2013a",)
    bibtex: tp.ClassVar[str] = """
        @misc{vaineau2018brain,
          doi = {10.5281/ZENODO.1494163},
          url = {https://zenodo.org/record/1494163},
          author = {Vaineau, Erwan and Barachant, Alexandre and Andreev, Anton and Rodrigues, Pedro L. C. and Cattan, Grégoire and Congedo, Marco},
          keywords = {Electroencephalography (EEG), P300, Brain-Computer Interface, Experiment, Adaptive, Calibration},
          language = {en},
          title = {Brain Invaders Adaptive versus Non-Adaptive P300 Brain-Computer Interface dataset},
          publisher = {Zenodo},
          year = {2018},
          copyright = {Creative Commons Attribution 1.0 Generic}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069"
    licence: tp.ClassVar[str] = "CC-BY-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 24 healthy participants while playing a P300-based Brain Invaders video game with 36 flashing symbols."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 33285, "NonTarget": 33286}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=73,
        num_subjects=24,
        num_events_in_query=481,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 139552),
        frequency=512.0,
    )


class Crell2024Handwritten(_BaseMoabb):
    """Subset of MOABB: BNCI2024_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2024_001",)
    bibtex: tp.ClassVar[str] = """
        @article{crell2024handwritten,
          title={Handwritten character classification from EEG through continuous kinematic decoding},
          volume={182},
          ISSN={0010-4825},
          url={http://dx.doi.org/10.1016/j.compbiomed.2024.109132},
          DOI={10.1016/j.compbiomed.2024.109132},
          journal={Computers in Biology and Medicine},
          publisher={Elsevier BV},
          author={Crell, Markus R. and Müller-Putz, Gernot R.},
          year={2024},
          month=nov,
          pages={109132}
        }
    """
    url: tp.ClassVar[str] = "http://bnci-horizon-2020.eu/database/data-sets/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 20 healthy participants while performing handwritten character motor imagery of 10 letters using the right index finger."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "letter_a": 1,
        "letter_d": 2,
        "letter_e": 3,
        "letter_f": 4,
        "letter_j": 5,
        "letter_n": 6,
        "letter_o": 7,
        "letter_s": 8,
        "letter_t": 9,
        "letter_v": 10,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=20,
        num_events_in_query=321,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 1536230),
        frequency=500.0,
    )


class Dornhege2004Boosting(_BaseMoabb):
    """Subset of MOABB: BNCI2003_004"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2003_004",)
    bibtex: tp.ClassVar[str] = """
        @article{dornhege2004boosting,
          title={Boosting bit rates in noninvasive EEG single-trial classifications by feature combination and multiclass paradigms},
          volume={51},
          ISSN={1558-2531},
          url={http://dx.doi.org/10.1109/TBME.2004.827088},
          DOI={10.1109/tbme.2004.827088},
          number={6},
          journal={IEEE Transactions on Biomedical Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Dornhege, G. and Blankertz, B. and Curio, G. and Muller, K.-R.},
          year={2004},
          month=jun,
          pages={993-1002}
        }
    """
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 5 healthy participants while performing right-hand and right-foot motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"right_hand": 0, "feet": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=5,
        num_subjects=5,
        num_events_in_query=169,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(118, 298458),
        frequency=100.0,
    )


class Dreyer2023Large(_BaseMoabb):
    """Subset of MOABB: Dreyer2023"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Dreyer2023",)
    bibtex: tp.ClassVar[str] = """
        @article{dreyer2023large,
          title={A large EEG database with users’ profile information for motor imagery brain-computer interface research},
          volume={10},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-023-02445-z},
          DOI={10.1038/s41597-023-02445-z},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Dreyer, Pauline and Roc, Aline and Pillette, Léa and Rimbert, Sébastien and Lotte, Fabien},
          year={2023},
          month=sep
        }
    """
    url: tp.ClassVar[str] = (
        "https://files.de-1.osf.io/v1/resources/8tdk5/providers/osfstorage/"
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (27-ch) in 87 healthy participants while performing left- and right-hand motor imagery across multiple BCI sessions."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=520,
        num_subjects=87,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(27, 230400),
        frequency=512.0,
    )


class Dreyer2023LargeA(_BaseMoabb):
    """Subset of MOABB: Dreyer2023A"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Dreyer2023A",)
    bibtex: tp.ClassVar[str] = """
        @article{dreyer2023large,
          title={A large EEG database with users’ profile information for motor imagery brain-computer interface research},
          volume={10},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-023-02445-z},
          DOI={10.1038/s41597-023-02445-z},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Dreyer, Pauline and Roc, Aline and Pillette, Léa and Rimbert, Sébastien and Lotte, Fabien},
          year={2023},
          month=sep
        }
    """
    url: tp.ClassVar[str] = (
        "https://files.de-1.osf.io/v1/resources/8tdk5/providers/osfstorage/"
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (5-ch) in 60 healthy participants while performing left- and right-hand motor imagery with visual feedback."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=358,
        num_subjects=60,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(27, 230400),
        frequency=512.0,
    )


class Dreyer2023LargeB(_BaseMoabb):
    """Subset of MOABB: Dreyer2023B"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Dreyer2023B",)
    bibtex: tp.ClassVar[str] = """
        @article{dreyer2023large,
          title={A large EEG database with users’ profile information for motor imagery brain-computer interface research},
          volume={10},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-023-02445-z},
          DOI={10.1038/s41597-023-02445-z},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Dreyer, Pauline and Roc, Aline and Pillette, Léa and Rimbert, Sébastien and Lotte, Fabien},
          year={2023},
          month=sep
        }
    """
    url: tp.ClassVar[str] = (
        "https://files.de-1.osf.io/v1/resources/8tdk5/providers/osfstorage/"
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 21 healthy participants while performing left- and right-hand motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=126,
        num_subjects=21,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(27, 210432),
        frequency=512.0,
    )


class Dreyer2023LargeC(_BaseMoabb):
    """Subset of MOABB: Dreyer2023C"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Dreyer2023C",)
    bibtex: tp.ClassVar[str] = """
        @article{dreyer2023large,
          title={A large EEG database with users’ profile information for motor imagery brain-computer interface research},
          volume={10},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-023-02445-z},
          DOI={10.1038/s41597-023-02445-z},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Dreyer, Pauline and Roc, Aline and Pillette, Léa and Rimbert, Sébastien and Lotte, Fabien},
          year={2023},
          month=sep
        }
    """
    url: tp.ClassVar[str] = (
        "https://files.de-1.osf.io/v1/resources/8tdk5/providers/osfstorage/"
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 6 healthy participants while performing left- and right-hand motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=36,
        num_subjects=6,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(27, 210432),
        frequency=512.0,
    )


class Faller2012Autocalibration(_BaseMoabb):
    """Subset of MOABB: BNCI2015_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_001",)
    bibtex: tp.ClassVar[str] = """
        @article{faller2012autocalibration,
          title={Autocalibration and Recurrent Adaptation: Towards a Plug and Play Online ERD-BCI},
          volume={20},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/tnsre.2012.2189584},
          DOI={10.1109/tnsre.2012.2189584},
          number={3},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Faller, Josef and Vidaurre, Carmen and Solis-Escalante, Teodoro and Neuper, Christa and Scherer, Reinhold},
          year={2012},
          month=may,
          pages={313-319}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (13-ch) in 12 healthy participants while performing sustained right-hand (palmar grasp) and feet (plantar extension) motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"right_hand": 1, "feet": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=28,
        num_subjects=12,
        num_events_in_query=201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(13, 1099541),
        frequency=512.0,
    )


class GrosseWentrup2009Beamforming(_BaseMoabb):
    """Subset of MOABB: GrosseWentrup2009"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("GrosseWentrup2009",)
    bibtex: tp.ClassVar[str] = """
        @article{grossewentrup2009beamforming,
          title={Beamforming in Noninvasive Brain-Computer Interfaces},
          volume={56},
          ISSN={1558-2531},
          url={http://dx.doi.org/10.1109/TBME.2008.2009768},
          DOI={10.1109/tbme.2008.2009768},
          number={4},
          journal={IEEE Transactions on Biomedical Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Grosse-Wentrup, M. and Liefhold, C. and Gramann, K. and Buss, M.},
          year={2009},
          month=apr,
          pages={1209-1219}
        }

        @misc{grossewentrup2018beamforming,
          author={Grosse-Wentrup, Moritz and Liefhold, Christian and Gramann, Klaus and Buss, Martin},
          title={Beamforming in Noninvasive Brain-Computer Interfaces Dataset},
          year={2018},
          publisher={Zenodo},
          doi={10.5281/zenodo.1217449},
          url={https://zenodo.org/records/1217449}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/1217449"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (128-ch) in 10 healthy participants while performing left- and right-hand motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"right_hand": 2, "left_hand": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=10,
        num_subjects=10,
        num_events_in_query=301,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(128, 1558340),
        frequency=500.0,
    )


class Guger2009How(_BaseMoabb):
    """Subset of MOABB: BNCI2015_003"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_003",)
    bibtex: tp.ClassVar[str] = """
        @article{guger2009how,
          title={How many people are able to control a P300-based brain-computer interface (BCI)?},
          volume={462},
          ISSN={0304-3940},
          url={http://dx.doi.org/10.1016/j.neulet.2009.06.045},
          DOI={10.1016/j.neulet.2009.06.045},
          number={1},
          journal={Neuroscience Letters},
          publisher={Elsevier BV},
          author={Guger, Christoph and Daban, Shahab and Sellers, Eric and Holzner, Clemens and Krausz, Gunther and Carabalona, Roberta and Gramatica, Furio and Edlinger, Guenter},
          year={2009},
          month=sep,
          pages={94-98}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (56-ch) in 10 healthy participants while attending to auditory spatial cues from six directions in an AMUSE P300 speller paradigm."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=20,
        num_subjects=10,
        num_events_in_query=2701,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 65282),
        frequency=256.0,
    )


def _load_haufe2011_raw(dl_path: Path, subject: int) -> mne.io.RawArray:
    """Load Haufe2011Eeg (BNCI2016-002) raw data directly from MATLAB files.

    Ensures ``marker_labels`` columns are correctly matched to
    ``event_mapping`` keys when creating annotations.
    """
    from moabb.datasets.bnci.bnci_2016_002 import _SUBJECT_VP_CODES, BBCI_URL
    from moabb.datasets.bnci.utils import bnci_data_path, convert_units, make_raw
    from pymatreader import read_mat

    vp_code = _SUBJECT_VP_CODES[subject]
    url = f"{BBCI_URL}VP{vp_code}.mat"
    filename = bnci_data_path(url, dl_path, False, None, None)[0]

    mat = read_mat(filename)
    sfreq = float(np.asarray(mat["cnt"]["fs"]).flat[0])
    ch_names = mat["cnt"]["clab"]
    if isinstance(ch_names, str):
        ch_names = [ch_names]
    data = np.asarray(mat["cnt"]["x"]).T
    marker_times = np.asarray(mat["mrk"]["time"]).flatten()
    marker_labels = np.asarray(mat["mrk"]["y"])

    eog_chs = {"EOGv", "EOGh"}
    emg_chs = {"EMGf"}
    misc_chs = {
        "lead_gas",
        "lead_brake",
        "dist_to_lead",
        "wheel_X",
        "wheel_Y",
        "gas",
        "brake",
    }
    ch_types = [
        (
            "eog"
            if ch in eog_chs
            else "emg"
            if ch in emg_chs
            else "misc"
            if ch in misc_chs
            else "eeg"
        )
        for ch in ch_names
    ]

    bio_mask = [i for i, ct in enumerate(ch_types) if ct in ("eeg", "eog", "emg")]
    data = convert_units(data.copy(), from_unit="uV", to_unit="V", channel_mask=bio_mask)

    raw = make_raw(
        data,
        ch_names,
        ch_types,
        sfreq,
        montage="standard_1005",
        line_freq=50.0,
        meas_date=datetime(2011, 1, 1, tzinfo=timezone.utc),
    )

    event_mapping = {0: "NonTarget", 1: "Target"}
    onset_times: list[float] = []
    descriptions: list[str] = []
    for i, time_ms in enumerate(marker_times):
        for class_idx, value in enumerate(marker_labels[:, i]):
            if value > 0 and class_idx in event_mapping:
                onset_times.append(time_ms / 1000.0)
                descriptions.append(event_mapping[class_idx])
                break

    if onset_times:
        raw.set_annotations(
            mne.Annotations(
                onset=onset_times,
                duration=[0.0] * len(onset_times),
                description=descriptions,
            )
        )

    return raw


class Haufe2011Eeg(_BaseMoabb):
    """Subset of MOABB: BNCI2016_002"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2016_002",)
    bibtex: tp.ClassVar[str] = """
        @article{haufe2011eeg,
          title={EEG potentials predict upcoming emergency brakings during simulated driving},
          volume={8},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2560/8/5/056001},
          DOI={10.1088/1741-2560/8/5/056001},
          number={5},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Haufe, Stefan and Treder, Matthias S and Gugler, Manfred F and Sagebaum, Max and Curio, Gabriel and Blankertz, Benjamin},
          year={2011},
          month=jul,
          pages={056001}
        }
    """
    url: tp.ClassVar[str] = (
        "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-EmergencyBraking/"
    )
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 15 participants while driving a simulated vehicle at 100 km/h and responding to emergency braking events."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=15,
        num_subjects=15,
        num_events_in_query=458,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(59, 1619936),
        frequency=200.0,
    )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        """Load raw data using custom MATLAB parser for correct marker handling."""
        tl = timeline
        dl_path = Path(self.path) / "download"
        subject = tl["subject"]

        key = (str(dl_path), subject)
        if key in _moabb_data_cache:
            _moabb_data_cache.move_to_end(key)
            return _moabb_data_cache[key][subject][tl["session"]][tl["run"]]

        raw = _load_haufe2011_raw(dl_path, subject)

        _moabb_data_cache[key] = {subject: {tl["session"]: {tl["run"]: raw}}}
        while len(_moabb_data_cache) > _MOABB_CACHE_MAXSIZE:
            _moabb_data_cache.popitem(last=False)

        return raw


class Hinss2021Open(_BaseMoabb):
    """Subset of MOABB: Hinss2021"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Hinss2021",)
    bibtex: tp.ClassVar[str] = """
        @inproceedings{hinss2021open,
          title={Open EEG datasets for passive brain-computer interface applications: Lacks and perspectives},
          author={Hinss, Marcel F and Somon, Bertille and Dehais, Fr{\'e}d{\'e}ric and Roy, Rapha{\"e}lle N},
          booktitle={2021 10th International IEEE/EMBS Conference on Neural Engineering (NER)},
          pages={686--689},
          year={2021},
          organization={IEEE},
          doi={10.1109/NER49283.2021.9441214}
        }

        @dataset{hinss2021eeg,
          title={An EEG dataset for cross-session mental workload estimation: Passive BCI competition of the Neuroergonomics Conference 2021},
          author={Hinss, Marcel F and Darmet, Ludovic and Somon, Bertille and Jahanpour, Emilie and Lotte, Fabien and Ladouce, Simon and Roy, Rapha{\"e}lle N},
          year={2021},
          publisher={Zenodo},
          doi={10.5281/zenodo.5055046},
          url={https://zenodo.org/records/5055046}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/5055046"
    licence: tp.ClassVar[str] = "CC-BY-SA-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 15 healthy participants while performing cognitive workload tasks (N-Back, MATB-II multi-tasking, vigilance, and Flanker)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"rs": 1, "easy": 2, "medium": 3, "diff": 4}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=30,
        num_subjects=15,
        num_events_in_query=478,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(61, 238500),
        frequency=500.0,
    )


class Hoffmann2008Efficient(_BaseMoabb):
    """Subset of MOABB: EPFLP300"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("EPFLP300",)
    bibtex: tp.ClassVar[str] = """
        @article{hoffmann2008efficient,
          title={An efficient P300-based brain-computer interface for disabled subjects},
          volume={167},
          ISSN={0165-0270},
          url={http://dx.doi.org/10.1016/j.jneumeth.2007.03.005},
          DOI={10.1016/j.jneumeth.2007.03.005},
          number={1},
          journal={Journal of Neuroscience Methods},
          publisher={Elsevier BV},
          author={Hoffmann, Ulrich and Vesin, Jean-Marc and Ebrahimi, Touradj and Diserens, Karin},
          year={2008},
          month=jan,
          pages={115-125}
        }
    """
    url: tp.ClassVar[str] = "http://documents.epfl.ch/groups/m/mm/mmspg/www/BCI/p300/"
    licence: tp.ClassVar[str] = "UNKNOWN"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 8 participants (healthy and motor-disabled) while performing a P300 BCI task with six visual stimulus categories."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=192,
        num_subjects=8,
        num_events_in_query=139,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 114687),
        frequency=2048.0,
    )


class Hubner2017Learning(_BaseMoabb):
    """Subset of MOABB: Huebner2017"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Huebner2017",)
    bibtex: tp.ClassVar[str] = """
        @article{hubner2017learning,
          title={Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees},
          volume={12},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0175856},
          DOI={10.1371/journal.pone.0175856},
          number={4},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Hübner, David and Verhoeven, Thibault and Schmid, Konstantin and Müller, Klaus-Robert and Tangermann, Michael and Kindermans, Pieter-Jan},
          editor={Congedo, Marco},
          year={2017},
          month=apr,
          pages={e0175856}
        }

        @misc{hubner2022raw,
          author={H\"{u}bner, David},
          title={Raw EEG Data for: Learning from Label Proportions in Brain-Computer Interfaces},
          year={2022},
          publisher={Zenodo},
          doi={10.5281/zenodo.5831826},
          url={https://zenodo.org/records/5831826}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/5831826"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (29-ch) in 13 healthy participants while copy-spelling with a visual P300 matrix speller (6x7 grid)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 10002, "NonTarget": 10001}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=342,
        num_subjects=13,
        num_events_in_query=477,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(31, 172570),
        frequency=1000.0,
    )


class Hubner2016Eeg(_BaseMoabb):
    """Subset of MOABB: Huebner2018"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Huebner2018",)
    bibtex: tp.ClassVar[str] = """
        @article{hubner2016eeg,
          title={Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees},
          author={Hübner, David and Verhoeven, Thibault and Müller, Klaus-Robert and Kindermans, Pieter-Jan and Tangermann, Michael},
          journal={IEEE Computational Intelligence Magazine},
          volume={13},
          number={2},
          pages={66--77},
          year={2018},
          doi={10.1109/MCI.2018.2807039},
          url={https://doi.org/10.1109/MCI.2018.2807039}
        }

        @dataset{hubner2016eegdata,
          doi={10.5281/zenodo.192684},
          url={https://zenodo.org/record/192684},
          author={Hübner, David},
          title={EEG Data For: "Learning From Label Proportions In Brain-Computer Interfaces"},
          publisher={Zenodo},
          year={2016}
        }
    """
    url: tp.ClassVar[str] = "https://doi.org/10.1109/MCI.2018.2807039"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (15-ch) in 12 healthy participants while performing a visual P300 matrix speller task with unsupervised label-proportion learning."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 10002, "NonTarget": 10001}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=360,
        num_subjects=12,
        num_events_in_query=477,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(31, 173137),
        frequency=1000.0,
    )


class Jao2023Eeg(_BaseMoabb):
    """Subset of MOABB: BNCI2022_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2022_001",)
    bibtex: tp.ClassVar[str] = """
        @article{jao2023eeg,
          title={EEG-Based Online Regulation of Difficulty in Simulated Flying},
          volume={14},
          ISSN={2371-9850},
          url={http://dx.doi.org/10.1109/TAFFC.2021.3059688},
          DOI={10.1109/taffc.2021.3059688},
          number={1},
          journal={IEEE Transactions on Affective Computing},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Jao, Ping-Keng and Chavarriaga, Ricardo and Millán, José del R.},
          year={2023},
          month=jan,
          pages={394-405}
        }
    """
    url: tp.ClassVar[str] = "http://bnci-horizon-2020.eu/database/data-sets/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 13 participants while piloting a simulated drone through waypoints at varying difficulty levels."
    )
    _DIFFICULTY_LABELS: tp.ClassVar[dict[int, str]] = {
        0: "easy",
        1: "hard",
        2: "extremely_hard",
    }
    event_id: tp.ClassVar[dict[str, int]] = {
        "easy": 0,
        "hard": 1,
        "extremely_hard": 2,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=13,
        num_subjects=13,
        num_events_in_query=33,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 1217280),
        frequency=256.0,
    )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        info = studies.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame([dict(type="Eeg", start="auto", filepath=info)])

        subject = timeline["subject"]
        mat_path = (
            Path(self.path)
            / "download"
            / "MNE-bnci-data"
            / "database"
            / "data-sets"
            / "001-2022"
            / f"s{subject}w.mat"
        )
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        subjective_report = mat["data"].subjective_report  # (32, 3)
        trigger = mat["data"].trigger
        sfreq = raw.info["sfreq"]

        # Detect trajectory boundaries via rising/falling edges
        traj_starts = np.where(np.diff((trigger == 1).astype(int)) == 1)[0] + 1
        traj_ends = np.where(np.diff((trigger == 255).astype(int)) == -1)[0] + 1

        n_traj = len(traj_starts)
        assert n_traj == subjective_report.shape[0], (
            f"Trajectory count mismatch: {n_traj} starts vs "
            f"{subjective_report.shape[0]} subjective reports"
        )

        rows: list[dict[str, tp.Any]] = []
        for i in range(n_traj):
            label = self._DIFFICULTY_LABELS[int(subjective_report[i, 2])]
            onset = traj_starts[i] / sfreq
            duration = (traj_ends[i] / sfreq) - onset if i < len(traj_ends) else 0.0
            rows.append(
                dict(
                    type="Stimulus",
                    start=onset,
                    duration=duration,
                    description=label,
                    code=self.event_id[label],
                )
            )

        events_df = pd.DataFrame(rows)
        return pd.concat([eeg_df, events_df]).reset_index(drop=True)


class Kalunga2016Online(_BaseMoabb):
    """Subset of MOABB: Kalunga2016"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Kalunga2016",)
    bibtex: tp.ClassVar[str] = """
        @article{kalunga2016online,
          title={Online SSVEP-based BCI using Riemannian geometry},
          volume={191},
          ISSN={0925-2312},
          url={http://dx.doi.org/10.1016/j.neucom.2016.01.007},
          DOI={10.1016/j.neucom.2016.01.007},
          journal={Neurocomputing},
          publisher={Elsevier BV},
          author={Kalunga, Emmanuel K. and Chevallier, Sylvain and Barthélemy, Quentin and Djouani, Karim and Monacelli, Eric and Hamam, Yskandar},
          year={2016},
          month=may,
          pages={55-68}
        }

        @misc{chevallier2015ssvep,
          author={Chevallier, Sylvain and Kalunga, Emmanuel},
          title={SSVEP-exoskeleton},
          year={2015},
          publisher={Zenodo},
          doi={10.5281/zenodo.2392979},
          url={https://zenodo.org/records/2392979}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2392979"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (8-ch) in 12 healthy participants while fixating on flickering visual stimuli in a 4-class SSVEP exoskeleton BCI task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"13": 2, "17": 4, "21": 3, "rest": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=30,
        num_subjects=12,
        num_events_in_query=33,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 57024),
        frequency=256.0,
    )


class Kappenman2021ErpErn(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_ERN"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_ERN",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while performing a flanker task eliciting error-related negativity (ERN) responses."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=401,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 935936),
        frequency=1024.0,
    )


class Kappenman2021ErpLrp(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_LRP"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_LRP",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while performing a flanker task eliciting lateralized readiness potentials (LRP)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=401,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 935936),
        frequency=1024.0,
    )


class Kappenman2021ErpMmn(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_MMN"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_MMN",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while listening to auditory oddball tone sequences eliciting mismatch negativity (MMN)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=1001,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 622592),
        frequency=1024.0,
    )


class Kappenman2021ErpN170(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_N170"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_N170",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while viewing faces and objects to elicit N170 face-selective potentials."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=321,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 699392),
        frequency=1024.0,
    )


class Kappenman2021ErpN2pc(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_N2pc"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_N2pc",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while performing a visual search task eliciting N2pc attention-related potentials."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=321,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 812032),
        frequency=1024.0,
    )


class Kappenman2021ErpP3(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_P3"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_P3",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while performing a visual oddball task eliciting P300 potentials."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 478208),
        frequency=1024.0,
    )


class Kappenman2021ErpN400(_BaseMoabb):
    """Subset of MOABB: ErpCore2021_N400"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("ErpCore2021_N400",)
    bibtex: tp.ClassVar[str] = """
        @article{kappenman2021erp,
          title={ERP CORE: An open resource for human event-related potential research},
          volume={225},
          ISSN={1053-8119},
          url={http://dx.doi.org/10.1016/j.neuroimage.2020.117465},
          DOI={10.1016/j.neuroimage.2020.117465},
          journal={NeuroImage},
          publisher={Elsevier BV},
          author={Kappenman, Emily S. and Farrens, Jaclyn L. and Zhang, Wendy and Stewart, Andrew X. and Luck, Steven J.},
          year={2021},
          month=jan,
          pages={117465}
        }
    """
    url: tp.ClassVar[str] = "https://files.osf.io/v1/resources/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 40 healthy participants while reading word pairs to elicit N400 semantic incongruity potentials."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=40,
        num_subjects=40,
        num_events_in_query=121,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 585728),
        frequency=1024.0,
    )


class Kojima2024ReplicationA(_BaseMoabb):
    """Subset of MOABB: Kojima2024A"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Kojima2024A",)
    bibtex: tp.ClassVar[str] = """
        @misc{kojima2024replication,
          doi = {10.7910/DVN/MQOVEY},
          url = {https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/MQOVEY},
          author = {Kojima, Simon and Kanoh, Shin'ichiro},
          keywords = {Engineering, Medicine, Health and Life Sciences, Electroencephalography, Event-related potentials, Brain-Computer Interface},
          title = {Replication Data for: An auditory brain-computer interface based on selective attention to multiple tone streams},
          publisher = {Harvard Dataverse},
          year = {2024}
        }
    """
    url: tp.ClassVar[str] = "https://dataverse.harvard.edu/api/access/datafile/"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 11 healthy participants while selectively attending to target tones among multiple auditory streams in a P300 BCI."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 0}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=66,
        num_subjects=11,
        num_events_in_query=175,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 308160),
        frequency=1000.0,
    )


class Kojima2024ReplicationB(_BaseMoabb):
    """Subset of MOABB: Kojima2024B"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Kojima2024B",)
    bibtex: tp.ClassVar[str] = """
        @misc{kojima2024replication,
          doi = {10.7910/DVN/1UJDV6},
          url = {https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/1UJDV6},
          author = {Kojima, Simon},
          keywords = {Engineering, Medicine, Health and Life Sciences, Brain-Computer Interface, Electroencephalography, Event-related potentials},
          title = {Replication Data for: Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing},
          publisher = {Harvard Dataverse},
          year = {2024}
        }
    """
    url: tp.ClassVar[str] = "https://dataverse.harvard.edu/api/access/datafile/"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 15 healthy participants while performing a four-class auditory selective-attention BCI task."
    )
    event_id: tp.ClassVar[dict[str, list[int]]] = {
        "Target": [111, 112, 113, 114],
        "NonTarget": [101, 102, 103, 104],
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=180,
        num_subjects=15,
        num_events_in_query=241,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 424520),
        frequency=1000.0,
    )


class Korczowski2019BrainBi2014A(_BaseMoabb):
    """Subset of MOABB: BI2014a"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2014a",)
    bibtex: tp.ClassVar[str] = """
        @misc{korczowski2019brain,
          title={Brain Invaders calibration-less P300-based BCI using dry EEG electrodes Dataset (BI2014a)},
          author={Korczowski, Louis and Ostaschenko, Ekaterina and Andreev, Anton and Cattan, Gr{\'e}goire and Coelho Rodrigues, Pedro Luis and Gautheret, Violette and Congedo, Marco},
          year={2019},
          publisher={Zenodo},
          doi={10.5281/zenodo.3266222},
          url={https://zenodo.org/record/3266222}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 64 healthy participants while playing a P300-based Brain Invaders game with 36 symbols and calibration-less dry electrodes."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=64,
        num_subjects=64,
        num_events_in_query=1189,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 417925),
        frequency=512.0,
    )


class Korczowski2019BrainBi2014B(_BaseMoabb):
    """Subset of MOABB: BI2014b"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2014b",)
    bibtex: tp.ClassVar[str] = """
        @misc{korczowski2019brain,
          title={Brain Invaders Solo versus Collaboration: Multi-User P300-based Brain-Computer Interface Dataset (BI2014b)},
          author={Korczowski, Louis and Ostaschenko, Ekaterina and Andreev, Anton and Cattan, Gr{\'e}goire and Coelho Rodrigues, Pedro Luis and Gautheret, Violette and Congedo, Marco},
          year={2019},
          publisher={Zenodo},
          doi={10.5281/zenodo.3267301},
          url={https://zenodo.org/record/3267301}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 38 healthy participants while playing a cooperative/competitive multi-user P300 Brain Invaders game."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=38,
        num_subjects=38,
        num_events_in_query=241,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 111680),
        frequency=512.0,
    )


class Korczowski2019BrainBi2015A(_BaseMoabb):
    """Subset of MOABB: BI2015a"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2015a",)
    bibtex: tp.ClassVar[str] = """
        @misc{korczowski2019brain,
          title={Brain Invaders calibration-less P300-based BCI with modulation of flash duration Dataset (BI2015a)},
          author={Korczowski, Louis and Cederhout, Martine and Andreev, Anton and Cattan, Gr{\'e}goire and Coelho Rodrigues, Pedro Luis and Gautheret, Violette and Congedo, Marco},
          year={2019},
          publisher={Zenodo},
          doi={10.5281/zenodo.3266929},
          url={https://zenodo.org/record/3266929}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 43 healthy participants while playing a P300 Brain Invaders game with modulated flash duration."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=129,
        num_subjects=43,
        num_events_in_query=2575,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 614400),
        frequency=512.0,
    )


class Korczowski2019BrainBi2015B(_BaseMoabb):
    """Subset of MOABB: BI2015b"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2015b",)
    bibtex: tp.ClassVar[str] = """
        @misc{korczowski2019brain,
          title={Brain Invaders Cooperative versus Competitive: Multi-User P300-based Brain-Computer Interface Dataset (BI2015b)},
          author={Korczowski, Louis and Cederhout, Martine and Andreev, Anton and Cattan, Gr{\'e}goire and Coelho Rodrigues, Pedro Luis and Gautheret, Violette and Congedo, Marco},
          year={2019},
          publisher={Zenodo},
          doi={10.5281/zenodo.3267307},
          url={https://zenodo.org/record/3267307}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 44 healthy participants while playing a cooperative/competitive multi-user P300 Brain Invaders game."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=176,
        num_subjects=44,
        num_events_in_query=661,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 285504),
        frequency=512.0,
    )


class Lee2019EegErp(_BaseMoabb):
    """Subset of MOABB: Lee2019_ERP"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Lee2019_ERP",)
    bibtex: tp.ClassVar[str] = """
        @article{lee2019eeg,
          title={EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy},
          volume={8},
          ISSN={2047-217X},
          url={http://dx.doi.org/10.1093/gigascience/giz002},
          DOI={10.1093/gigascience/giz002},
          number={5},
          journal={GigaScience},
          publisher={Oxford University Press (OUP)},
          author={Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
          year={2019},
          month=jan
        }
    """
    url: tp.ClassVar[str] = (
        "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"
    )
    licence: tp.ClassVar[str] = "GPL-3.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (62-ch) in 54 healthy participants while performing a 36-character P300 row-column speller with face-overlay stimuli."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=108,
        num_subjects=54,
        num_events_in_query=1981,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(62, 1143560),
        frequency=1000.0,
    )


class Lee2019EegMi(_BaseMoabb):
    """Subset of MOABB: Lee2019_MI"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Lee2019_MI",)
    bibtex: tp.ClassVar[str] = """
        @article{lee2019eeg,
          title={EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy},
          volume={8},
          ISSN={2047-217X},
          url={http://dx.doi.org/10.1093/gigascience/giz002},
          DOI={10.1093/gigascience/giz002},
          number={5},
          journal={GigaScience},
          publisher={Oxford University Press (OUP)},
          author={Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
          year={2019},
          month=jan
        }
    """
    url: tp.ClassVar[str] = (
        "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"
    )
    licence: tp.ClassVar[str] = "GPL-3.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (62-ch) in 54 healthy participants while performing left- and right-hand motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 2, "right_hand": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=54,
        num_subjects=54,
        num_events_in_query=101,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(62, 1519160),
        frequency=1000.0,
    )


class Lee2019EegSsvep(_BaseMoabb):
    """Subset of MOABB: Lee2019_SSVEP"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Lee2019_SSVEP",)
    bibtex: tp.ClassVar[str] = """
        @article{lee2019eeg,
          title={EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy},
          volume={8},
          ISSN={2047-217X},
          url={http://dx.doi.org/10.1093/gigascience/giz002},
          DOI={10.1093/gigascience/giz002},
          number={5},
          journal={GigaScience},
          publisher={Oxford University Press (OUP)},
          author={Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
          year={2019},
          month=jan
        }
    """
    url: tp.ClassVar[str] = (
        "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"
    )
    licence: tp.ClassVar[str] = "GPL-3.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (62-ch) in 54 healthy participants while fixating on SSVEP targets at four frequencies (5.45, 6.67, 8.57, 12 Hz)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"12.0": 1, "8.57": 2, "6.67": 3, "5.45": 4}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=54,
        num_subjects=54,
        num_events_in_query=101,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(62, 1298000),
        frequency=1000.0,
    )


class Leeb2007Brain(_BaseMoabb):
    """Subset of MOABB: BNCI2014_004"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2014_004",)
    bibtex: tp.ClassVar[str] = """
        @article{leeb2007brain,
          title={Brain-Computer Communication: Motivation, Aim, and Impact of Exploring a Virtual Apartment},
          volume={15},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/TNSRE.2007.906956},
          DOI={10.1109/tnsre.2007.906956},
          number={4},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Leeb, Robert and Lee, Felix and Keinrath, Claudia and Scherer, Reinhold and Bischof, Horst and Pfurtscheller, Gert},
          year={2007},
          month=dec,
          pages={473-482}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/"
    licence: tp.ClassVar[str] = "CC-BY-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (22-ch) in 9 healthy participants while performing four-class motor imagery (left hand, right hand, feet, tongue)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=45,
        num_subjects=9,
        num_events_in_query=121,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(3, 604803),
        frequency=250.0,
    )


class Liu2024Eeg(_BaseMoabb):
    """Subset of MOABB: Liu2024"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Liu2024",)
    bibtex: tp.ClassVar[str] = """
        @article{liu2024eeg,
          title={An EEG motor imagery dataset for brain computer interface in acute stroke patients},
          volume={11},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-023-02787-8},
          DOI={10.1038/s41597-023-02787-8},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Liu, Haijie and Wei, Penghu and Wang, Haochong and Lv, Xiaodong and Duan, Wei and Li, Meijie and Zhao, Yan and Wang, Qingmei and Chen, Xinyuan and Shi, Gaige and Han, Bo and Hao, Junwei},
          year={2024},
          month=jan
        }
    """
    url: tp.ClassVar[str] = "https://ndownloader.figshare.com/files/38516078"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (30-ch) in 50 participants with acute stroke while imagining grasping a spherical object with the left or right hand."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=50,
        num_subjects=50,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(29, 160000),
        frequency=500.0,
    )


def _martinez_cagigal_convert_to_mne_format(  # type: ignore[no-untyped-def]
    self, path, true_labels=None
):
    """Patched ``_convert_to_mne_format`` for MartinezCagigal2023Checker and MartinezCagigal2023Pary.

    Upstream MOABB writes ``np.unique(trial_idx)`` (per-recording trial index
    ``0..n_trials-1``) into ``stim_trial`` for these two datasets, instead
    of the attended command id. That breaks downstream multiclass
    classification across recordings as the same marker corresponds to
    different attended commands. Every other c-VEP dataset in MOABB
    (``CabreraCastillos2023*``, ``Thielen2015``, ``Thielen2021``) writes the
    attended-symbol id, so this patch brings the Martinez-Cagigal datasets
    in line with the rest of the c-VEP collection.

    Implementation: call the original upstream method, then post-process
    the resulting ``Raw`` to (1) replace ``stim_trial`` markers with
    ``command_id + 200`` and (2) augment the existing ``_trial_meta``
    annotation extras with a ``command_id`` field (preserving ``trial_id``).
    The ``stim_epoch`` channel is left untouched: its codebook columns are
    indexed by the per-recording ``trial_idx``, which is correct.
    """
    from moabb.datasets.utils import (  # type: ignore[import-untyped,import-not-found]
        add_stim_channel_trial,
    )

    raw = type(self)._upstream_convert_to_mne_format(self, path, true_labels)

    cvep_data = self._trim_unfinished_trial(
        self._load_bson_recording(path)["cvepspellerdata"]
    )
    trial_idx_arr = np.array(cvep_data["trial_idx"], dtype=int)
    unique_trials = np.unique(trial_idx_arr)

    if cvep_data["mode"] == "train":
        cmd_arr = np.array(cvep_data["command_idx"], dtype=int)
        command_ids = np.array(
            [int(cmd_arr[np.where(trial_idx_arr == t)[0][0]]) for t in unique_trials],
            dtype=int,
        )
    else:
        assert true_labels is not None
        label_to_cmd = {
            item["label"]: int(c) for c, item in cvep_data["commands_info"][0].items()
        }
        command_ids = np.array(
            [label_to_cmd[true_labels[int(t)]] for t in unique_trials], dtype=int
        )

    trial_events = mne.find_events(
        raw, stim_channel="stim_trial", shortest_event=0, verbose=False
    )
    assert len(trial_events) == len(command_ids), (
        f"stim_trial has {len(trial_events)} events but BSON has "
        f"{len(command_ids)} trials"
    )
    raw.drop_channels(["stim_trial"])
    raw = add_stim_channel_trial(raw, trial_events[:, 0], command_ids, offset=200)

    raw.annotations.extras = [
        {**(extra or {}), "command_id": int(cid)}
        for extra, cid in zip(raw.annotations.extras, command_ids)
    ]
    return raw


_martinez_cagigal_convert_to_mne_format._neuralhub_command_id_patch = True  # type: ignore[attr-defined]


class _MartinezCagigalCvepBase(_BaseCvepTrialMoabb):
    """Shared base for the two Martinez-Cagigal 2023 c-VEP datasets.

    Both upstream MOABB classes (``MartinezCagigal2023Checker`` and
    ``MartinezCagigal2023Pary``) write per-recording trial indices into
    the ``stim_trial`` channel instead of the attended command id. This
    base class lazily replaces ``_convert_to_mne_format`` on the upstream
    class the first time a Martinez-Cagigal recording is loaded, so the
    fix is owned by the dataset that needs it (no import-time side
    effects on the rest of the module).

    The replacement is implemented in
    :func:`_martinez_cagigal_convert_to_mne_format`; the patch is
    idempotent (guarded by a sentinel attribute on the bound method).
    """

    _presentation_rate: tp.ClassVar[float] = 120.0
    _moabb_patched: tp.ClassVar[bool] = False

    @classmethod
    def _ensure_moabb_patched(cls) -> None:
        if _MartinezCagigalCvepBase._moabb_patched:
            return
        try:
            from moabb.datasets import (  # type: ignore[import-untyped,import-not-found]
                martinezcagigal2023_checker_cvep as _checker_mod,
            )
            from moabb.datasets import (
                martinezcagigal2023_pary_cvep as _pary_mod,
            )
        except ImportError:
            return
        for mod, cls_name in (
            (_checker_mod, "MartinezCagigal2023Checker"),
            (_pary_mod, "MartinezCagigal2023Pary"),
        ):
            upstream_cls = getattr(mod, cls_name, None)
            if upstream_cls is None:
                continue
            existing = getattr(upstream_cls, "_convert_to_mne_format", None)
            if existing is None or getattr(
                existing, "_neuralhub_command_id_patch", False
            ):
                continue
            # Save the original so the patched method can call back into it.
            upstream_cls._upstream_convert_to_mne_format = existing
            upstream_cls._convert_to_mne_format = _martinez_cagigal_convert_to_mne_format
        _MartinezCagigalCvepBase._moabb_patched = True

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        type(self)._ensure_moabb_patched()
        return super()._load_raw(timeline)


class MartinezCagigal2023Dataset(_MartinezCagigalCvepBase):
    """Subset of MOABB: MartinezCagigal2023Checker"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("MartinezCagigal2023Checker",)
    # Cycle-level splitting: 8 repeats of a 63-bit binary m-sequence per trial,
    # displayed at 120 Hz (cycle_dur = 63/120 = 0.525 s). All 8 spatial-freq
    # conditions share the same code, so a single class-level value is enough.
    _code_length: tp.ClassVar[int] = 63
    _n_cycles: tp.ClassVar[int] = 8
    bibtex: tp.ClassVar[str] = """
        @misc{martinezcagigal2023dataset,
          doi = {10.71569/7C67-V596},
          url = {https://uvadoc.uva.es/handle/10324/70973},
          author = {Martínez-Cagigal, Víctor},
          keywords = {Brain-computer interface (BCI), Code-modulated visual evoked potential (c-VEP), Electroencephalography (EEG), Visual fatigue, Spatial Frequency},
          language = {en},
          title = {Dataset: Influence of spatial frequency in visual stimuli for cVEP-based BCIs: evaluation of performance and user experience},
          publisher = {Universidad de Valladolid},
          year = {2023},
          copyright = {Creative Commons Attribution Non Commercial 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://uvadoc.uva.es/bitstream/handle/10324/70973"
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 16 healthy participants while performing a checkerboard m-sequence c-VEP BCI speller task."
    )
    # 9-command (3x3) speller; 8 spatial-frequency conditions share the
    # same alphabet. `_MartinezCagigalCvepBase` patches the upstream MOABB
    # class so each `stim_trial` marker encodes the attended command id.
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(9)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=383,
        num_subjects=16,
        num_events_in_query=121,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 29712),
        frequency=256.0,
    )


class MartinezCagigal2024Dataset(_MartinezCagigalCvepBase):
    """Subset of MOABB: MartinezCagigal2023Pary"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("MartinezCagigal2023Pary",)
    # Cycle-level splitting: 10 repeats of a p-ary m-sequence per trial, at
    # 120 Hz. The m-sequence length depends on the base (one base per session,
    # session names follow the ``"<i>base<p>"`` format from MOABB).
    # Lengths: GF(2^6)=63, GF(3^4)=80, GF(5^3)=124, GF(7^2)=48, GF(11^2)=120.
    _CODE_LENGTH_BY_BASE: tp.ClassVar[dict[str, int]] = {
        "2": 63,
        "3": 80,
        "5": 124,
        "7": 48,
        "11": 120,
    }
    _n_cycles: tp.ClassVar[int] = 10

    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=600,
        num_subjects=15,
        num_events_in_query=51,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 11736),
        frequency=256.0,
    )

    def _get_code_length(self, timeline: dict[str, tp.Any]) -> int | None:
        session = str(timeline["session"])
        match = re.match(r"\d+base(\d+)", session)
        if match is None:
            raise ValueError(
                f"{type(self).__name__}: cannot parse base from session "
                f"name {session!r} (expected '<i>base<p>')."
            )
        return self._CODE_LENGTH_BY_BASE[match.group(1)]

    bibtex: tp.ClassVar[str] = """
        @misc{martinezcagigal2024dataset,
          doi = {10.71569/025S-EQ10},
          url = {https://uvadoc.uva.es/handle/10324/70945},
          author = {Martínez-Cagigal, Víctor},
          keywords = {Non-binary codes, Visual fatigue, Code-modulated visual evoked potential (c-VEP), Brain-computer interface (BCI), Electroencephalography (EEG)},
          language = {en},
          title = {Dataset: Non-binary m-sequences for more comfortable brain-computer interfaces based on c-VEPs},
          publisher = {Universidad de Valladolid},
          year = {2024},
          copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://uvadoc.uva.es/bitstream/handle/10324/70945"
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 16 healthy participants while performing a non-binary (p-ary) m-sequence c-VEP BCI speller task."
    )
    # 16-command speller; 5 p-ary m-sequence bases (2/3/5/7/11) share the
    # same alphabet. `_MartinezCagigalCvepBase` patches the upstream MOABB
    # class so each `stim_trial` marker encodes the attended command id.
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(16)}


class Nakanishi2015Comparison(_BaseMoabb):
    """Subset of MOABB: Nakanishi2015"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Nakanishi2015",)
    bibtex: tp.ClassVar[str] = """
        @article{nakanishi2015comparison,
          title={A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials},
          volume={10},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0140703},
          DOI={10.1371/journal.pone.0140703},
          number={10},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Nakanishi, Masaki and Wang, Yijun and Wang, Yu-Te and Jung, Tzyy-Ping},
          editor={Yao, Dezhong},
          year={2015},
          month=oct,
          pages={e0140703}
        }
    """
    url: tp.ClassVar[str] = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"
    licence: tp.ClassVar[str] = "UNKNOWN"
    description: tp.ClassVar[str] = (
        "EEG recordings (8-ch) in 9 healthy participants while fixating on flickering stimuli in a 12-class SSVEP target identification task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "9.25": 1,
        "11.25": 2,
        "13.25": 3,
        "9.75": 4,
        "11.75": 5,
        "13.75": 6,
        "10.25": 7,
        "12.25": 8,
        "14.25": 9,
        "10.75": 10,
        "12.75": 11,
        "14.75": 12,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=9,
        num_subjects=9,
        num_events_in_query=181,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 218520),
        frequency=256.0,
    )


class Ofner2017Upper(_BaseMoabb):
    """Subset of MOABB: Ofner2017"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Ofner2017",)
    bibtex: tp.ClassVar[str] = """
        @article{ofner2017upper,
          title={Upper limb movements can be decoded from the time-domain of low-frequency EEG},
          volume={12},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0182578},
          DOI={10.1371/journal.pone.0182578},
          number={8},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Ofner, Patrick and Schwarz, Andreas and Pereira, Joana and Müller-Putz, Gernot R.},
          editor={Zhang, Dingguo},
          year={2017},
          month=aug,
          pages={e0182578}
        }

        @misc{ofner2017upperb,
          author={Ofner, Patrick and Schwarz, Andreas and Pereira, Joana and M\"{u}ller-Putz, Gernot R.},
          title={Upper limb movements can be decoded from the time-domain of low-frequency EEG},
          year={2017},
          publisher={Zenodo},
          doi={10.5281/zenodo.834976},
          url={https://zenodo.org/records/834976}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/834976"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (61-ch) in 15 healthy participants while performing and imagining upper-limb movements (elbow, forearm, hand open/close)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "right_elbow_flexion": 1536,
        "right_elbow_extension": 1537,
        "right_supination": 1538,
        "right_pronation": 1539,
        "right_hand_close": 1540,
        "right_hand_open": 1541,
        "rest": 1542,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=150,
        num_subjects=15,
        num_events_in_query=43,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(61, 165888),
        frequency=512.0,
    )


class Ofner2017UpperExecution(Ofner2017Upper):
    """Subset of MOABB: Ofner2017 (execution session only)

    Same underlying dataset as Ofner2017Upper but loads the motor execution
    session instead of the motor imagery session.
    """

    _dataset_kwargs: tp.ClassVar[dict[str, tp.Any]] = {
        "imagined": False,
        "executed": True,
    }
    description: tp.ClassVar[str] = (
        "EEG recordings (61-ch) in 15 healthy participants while executing upper-limb movements (elbow, forearm, hand open/close)."
    )
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=150,
        num_subjects=15,
        num_events_in_query=43,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(61, 164864),
        frequency=512.0,
    )


class Ofner2019Attempted(_BaseMoabb):
    """Subset of MOABB: BNCI2019_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2019_001",)
    bibtex: tp.ClassVar[str] = """
        @article{ofner2019attempted,
          title={Attempted Arm and Hand Movements can be Decoded from Low-Frequency EEG from Persons with Spinal Cord Injury},
          volume={9},
          ISSN={2045-2322},
          url={http://dx.doi.org/10.1038/s41598-019-43594-9},
          DOI={10.1038/s41598-019-43594-9},
          number={1},
          journal={Scientific Reports},
          publisher={Springer Science and Business Media LLC},
          author={Ofner, Patrick and Schwarz, Andreas and Pereira, Joana and Wyss, Daniela and Wildburger, Renate and Müller-Putz, Gernot R.},
          year={2019},
          month=may
        }
    """
    url: tp.ClassVar[str] = "http://bnci-horizon-2020.eu/database/data-sets/001-2019/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (61-ch) in 10 participants with spinal cord injury while attempting arm and hand movements."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "supination": 776,
        "pronation": 777,
        "hand_open": 779,
        "palmar_grasp": 925,
        "lateral_grasp": 926,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=90,
        num_subjects=10,
        num_events_in_query=41,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(61, 75520),
        frequency=256.0,
    )


class Oikonomou2016ComparativeA(_BaseMoabb):
    """Subset of MOABB: MAMEM1"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("MAMEM1",)
    bibtex: tp.ClassVar[str] = """
        @misc{oikonomou2016comparative,
          doi = {10.48550/ARXIV.1602.00904},
          url = {https://arxiv.org/abs/1602.00904},
          author = {Oikonomou, Vangelis P. and Liaros, Georgios and Georgiadis, Kostantinos and Chatzilari, Elisavet and Adam, Katerina and Nikolopoulos, Spiros and Kompatsiaris, Ioannis},
          keywords = {Human-Computer Interaction (cs.HC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
          title = {Comparative evaluation of state-of-the-art algorithms for SSVEP-based BCIs},
          publisher = {arXiv},
          year = {2016},
          copyright = {arXiv.org perpetual, non-exclusive license}
        }
    """
    url: tp.ClassVar[str] = "https://figshare.com/articles/dataset/2068677"
    licence: tp.ClassVar[str] = "ODC-By-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (256-ch) in 11 healthy participants while fixating on flickering visual stimuli in a 5-class SSVEP task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "6.66": 1,
        "7.50": 2,
        "8.57": 3,
        "10.00": 4,
        "12.00": 5,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=47,
        num_subjects=11,
        num_events_in_query=24,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(256, 117917),
        frequency=250.0,
    )


class Oikonomou2016ComparativeB(_BaseMoabb):
    """Subset of MOABB: MAMEM2"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("MAMEM2",)
    bibtex: tp.ClassVar[str] = """
        @misc{oikonomou2016comparative,
          doi = {10.48550/ARXIV.1602.00904},
          url = {https://arxiv.org/abs/1602.00904},
          author = {Oikonomou, Vangelis P. and Liaros, Georgios and Georgiadis, Kostantinos and Chatzilari, Elisavet and Adam, Katerina and Nikolopoulos, Spiros and Kompatsiaris, Ioannis},
          keywords = {Human-Computer Interaction (cs.HC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
          title = {Comparative evaluation of state-of-the-art algorithms for SSVEP-based BCIs},
          publisher = {arXiv},
          year = {2016},
          copyright = {arXiv.org perpetual, non-exclusive license}
        }
    """
    url: tp.ClassVar[str] = "https://figshare.com/articles/dataset/3153409"
    licence: tp.ClassVar[str] = "ODC-By-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (256-ch) in 11 healthy participants while fixating on simultaneously presented flickering stimuli in a 5-class SSVEP task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "6.66": 1,
        "7.50": 2,
        "8.57": 3,
        "10.00": 4,
        "12.00": 5,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=55,
        num_subjects=11,
        num_events_in_query=26,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(256, 81690),
        frequency=250.0,
    )


class Oikonomou2016ComparativeC(_BaseMoabb):
    """Subset of MOABB: MAMEM3"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("MAMEM3",)
    bibtex: tp.ClassVar[str] = """
        @misc{oikonomou2016comparative,
          doi = {10.48550/ARXIV.1602.00904},
          url = {https://arxiv.org/abs/1602.00904},
          author = {Oikonomou, Vangelis P. and Liaros, Georgios and Georgiadis, Kostantinos and Chatzilari, Elisavet and Adam, Katerina and Nikolopoulos, Spiros and Kompatsiaris, Ioannis},
          keywords = {Human-Computer Interaction (cs.HC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
          title = {Comparative evaluation of state-of-the-art algorithms for SSVEP-based BCIs},
          publisher = {arXiv},
          year = {2016},
          copyright = {arXiv.org perpetual, non-exclusive license}
        }
    """
    url: tp.ClassVar[str] = "https://figshare.com/articles/dataset/3413851"
    licence: tp.ClassVar[str] = "ODC-By-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (14-ch) in 11 healthy participants while fixating on simultaneously presented flickering stimuli in a 5-class SSVEP task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "6.66": 33029,
        "7.50": 33028,
        "8.57": 33027,
        "10.00": 33026,
        "12.00": 33025,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=110,
        num_subjects=11,
        num_events_in_query=13,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(14, 17792),
        frequency=128.0,
    )


class Pulferer2022Continuous(_BaseMoabb):
    """Subset of MOABB: BNCI2025_002"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2025_002",)
    bibtex: tp.ClassVar[str] = """
        @article{pulferer2022continuous,
          title={Continuous 2D trajectory decoding from attempted movement: across-session performance in able-bodied and feasibility in a spinal cord injured participant},
          volume={19},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2552/ac689f},
          DOI={10.1088/1741-2552/ac689f},
          number={3},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Pulferer, Hannah S and Ásgeirsdóttir, Brynja and Mondini, Valeria and Sburlea, Andreea I and Müller-Putz, Gernot R},
          year={2022},
          month=may,
          pages={036005}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/001-2025/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (60-ch) in 10 healthy participants while attempting continuous 2D reaching movements with trajectory visual feedback."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"snakerun": 1, "freerun": 2, "eyerun": 3}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=90,
        num_subjects=10,
        num_events_in_query=49,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 273696),
        frequency=200.0,
    )


class Reichert2020Impact(_BaseMoabb):
    """Subset of MOABB: BNCI2020_002.

    MOABB only emits one Stimulus annotation per 16-second trial (at trial
    onset), which is unusable for a stimulus-locked N2pc analysis.  The raw
    ``.mat`` file, however, contains a per-sample 2-channel ``bciexp.stim``
    array encoding the exact onset of every "green-on-left" (``stim[0]``)
    and "green-on-right" (``stim[1]``) display, together with a per-trial
    ``bciexp.intention`` string ("yes"/"no") indicating which color the
    participant attended (yes = green, no = red).

    We override :meth:`_load_timeline_events` to reconstruct per-stimulus
    events directly from ``bciexp.stim``: each trial yields 10 stimulus
    events labelled ``Target`` if the participant's attended hemifield is
    contralateral to ``PO7`` (i.e. attended-right) and ``NonTarget``
    otherwise.  This brings the per-subject event count from 120 to ~1200
    and restores the canonical N2pc trial structure.

    We also override :meth:`_load_raw` to fix a layout bug in MOABB's
    ``BNCI2020_002._convert_attention_shift`` (in
    ``moabb/datasets/bnci/bnci_2020.py``).  The raw EEG comes out of
    ``scipy.io.loadmat`` as an F-contiguous array of shape
    ``(n_channels, n_samples_per_trial, n_trials)``, and MOABB flattens it
    to a continuous ``(n_channels, n_samples_per_trial * n_trials)`` array
    via ``bciexp.data.reshape(n_channels, -1)``.  Because the source array
    is F-contiguous, the default C-order reshape produces a *trial-fastest*
    layout (``flat[c, k] = data[c, k // n_trials, k % n_trials]``) instead
    of the trial-major layout that MOABB's per-trial markers and our
    per-stimulus events both assume.  The correct reshape is
    ``data.transpose(0, 2, 1).reshape(n_channels, -1)``.  Without this
    patch, every per-stimulus epoch is sampled from the wrong trial and
    the wrong sample offset, the canonical N2pc is invisible, and
    per-stim Target/NonTarget decoding sits at chance.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2020_002",)
    bibtex: tp.ClassVar[str] = """
        @article{reichert2020impact,
          title={Impact of Stimulus Features on the Performance of a Gaze-Independent Brain-Computer Interface Based on Covert Spatial Attention Shifts},
          volume={14},
          ISSN={1662-453X},
          url={http://dx.doi.org/10.3389/fnins.2020.591777},
          DOI={10.3389/fnins.2020.591777},
          journal={Frontiers in Neuroscience},
          publisher={Frontiers Media SA},
          author={Reichert, Christoph and Tellez Ceja, Igor Fabian and Sweeney-Reed, Catherine M. and Heinze, Hans-Jochen and Hinrichs, Hermann and Dürschmid, Stefan},
          year={2020},
          month=dec
        }
    """
    url: tp.ClassVar[str] = "http://bnci-horizon-2020.eu/database/data-sets/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (30-ch) in 18 healthy participants while performing a covert spatial attention N2pc task to answer yes/no questions."  # codespell:ignore covert
    )
    event_id: tp.ClassVar[dict[str, int]] = {"NonTarget": 1, "Target": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=18,
        num_subjects=18,
        num_events_in_query=1201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 480000),
        frequency=250.0,
    )

    _TARGET_TOKENS: tp.ClassVar[frozenset[str]] = frozenset(
        {"yes", "y", "right", "r", "target", "true"}
    )

    def _mat_path(self, timeline: dict[str, tp.Any]) -> Path:
        """Return the path to ``P<subject>.mat`` on disk."""
        subject = int(timeline["subject"])
        return (
            Path(self.path)
            / "download"
            / "MNE-bnci-data"
            / "database"
            / "data-sets"
            / "002-2020"
            / f"P{subject:02d}.mat"
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raw = super()._load_raw(timeline)

        # Re-load the raw EEG from the .mat with the correct layout to
        # work around the MOABB BNCI2020_002 reshape bug.  See class
        # docstring for details.
        mat = loadmat(
            str(self._mat_path(timeline)), struct_as_record=False, squeeze_me=True
        )
        bciexp = mat["bciexp"]
        data = np.asarray(bciexp.data)  # (n_channels, n_samples, n_trials)
        n_channels, n_samples, n_trials = data.shape
        eeg_correct = (
            data.transpose(0, 2, 1)
            .reshape(n_channels, n_samples * n_trials)
            .astype(raw._data.dtype)
        )
        # MOABB's `_convert_attention_shift` scales raw .mat data (uV) to
        # V via ``* 1e-6``; mirror that scaling here so the patched
        # channels keep MOABB's unit convention.
        eeg_correct *= 1e-6

        # Replace the EEG channels in MOABB's Raw.  MOABB lays out
        # ``ch_names_full = list(bciexp.label) + ["HEOG", "VEOG", "STI"]``,
        # so the first ``n_channels`` rows of ``raw._data`` are the EEG.
        # HEOG/VEOG are flattened via ``arr.T.reshape(1, -1)`` which is
        # already correct (transpose makes the F-contiguous (samples,
        # trials) array C-contiguous before reshape), so we leave them
        # untouched.  The STI channel is built sample-by-sample at
        # ``trial_idx * n_samples`` -- consistent with trial-major
        # layout, so it is also correct *after* this EEG fix.
        raw._data[:n_channels] = eeg_correct  # noqa: SLF001
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = studies.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame([dict(type="Eeg", start="auto", filepath=info)])

        mat = loadmat(
            str(self._mat_path(timeline)), struct_as_record=False, squeeze_me=True
        )
        bciexp = mat["bciexp"]
        sfreq = float(bciexp.srate)
        stim = np.asarray(bciexp.stim)  # (2, n_samples_per_trial, n_trials)
        intention = np.asarray(bciexp.intention)
        n_samples_per_trial = int(stim.shape[1])
        n_trials = int(stim.shape[2])

        rows: list[dict[str, tp.Any]] = []
        for trial_idx in range(n_trials):
            token = str(intention[trial_idx]).strip().lower()
            # ``stim[0]`` fires for "green-on-left" displays and ``stim[1]``
            # for "green-on-right" displays (verified against
            # ``bciexp.targetside``).  In a yes-trial the participant
            # attends green, so right-side displays (``stim[1]``) become
            # Target = attended-right; in a no-trial they attend red, so
            # the left-side green display means red is on the right and
            # ``stim[0]`` becomes Target = attended-right.
            if token in self._TARGET_TOKENS:
                target_ch, nontarget_ch = 1, 0
            else:
                target_ch, nontarget_ch = 0, 1
            for ch_idx, label in ((target_ch, "Target"), (nontarget_ch, "NonTarget")):
                ch = stim[ch_idx, :, trial_idx].astype(np.int32)
                edges = np.where(np.diff(ch) > 0)[0] + 1
                for edge in edges:
                    onset_samp = trial_idx * n_samples_per_trial + int(edge)
                    rows.append(
                        {
                            "start": onset_samp / sfreq,
                            "duration": 0.0,
                            "description": label,
                            "code": self.event_id[label],
                        }
                    )

        events_df = pd.DataFrame(rows)
        events_df.insert(0, "type", "Stimulus")
        result = pd.concat([eeg_df, events_df]).reset_index(drop=True)
        result["description"] = result["description"].fillna("").astype(str)
        return result


class Riccio2013Attention(_BaseMoabb):
    """Subset of MOABB: BNCI2014_008"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2014_008",)
    bibtex: tp.ClassVar[str] = """
        @article{riccio2013attention,
          title={Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis},
          volume={7},
          ISSN={1662-5161},
          url={http://dx.doi.org/10.3389/fnhum.2013.00732},
          DOI={10.3389/fnhum.2013.00732},
          journal={Frontiers in Human Neuroscience},
          publisher={Frontiers Media SA},
          author={Riccio, Angela and Simione, Luca and Schettini, Francesca and Pizzimenti, Alessia and Inghilleri, Maurizio and Belardinelli, Marta Olivetti and Mattia, Donatella and Cincotti, Febo},
          year={2013}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (8-ch) in 8 participants with ALS while performing a visual P300 BCI speller task."  # codespell:ignore als
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=8,
        num_subjects=8,
        num_events_in_query=4201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 347704),
        frequency=256.0,
    )


class Cattan2018Eeg(_BaseMoabb):
    """Subset of MOABB: Rodrigues2017"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Rodrigues2017",)
    bibtex: tp.ClassVar[str] = """
        @misc{cattan2018eeg,
          doi = {10.5281/ZENODO.2348891},
          url = {https://zenodo.org/record/2348891},
          author = {Cattan, Grégoire and Rodrigues, Pedro L. C. and Congedo, Marco},
          title = {EEG Alpha Waves dataset},
          publisher = {Zenodo},
          year = {2018},
          copyright = {Creative Commons Attribution 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2348892/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 19 healthy participants while alternating between eyes-closed and eyes-open resting-state conditions."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"closed": 1, "open": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=19,
        num_subjects=19,
        num_events_in_query=11,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 119808),
        frequency=512.0,
    )


class Romani2025Brainform(_BaseMoabb):
    """Subset of MOABB: RomaniBF2025ERP

    Subjects P15 and P18 are excluded by MOABB (did not complete the full
    protocol).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("RomaniBF2025ERP",)
    bibtex: tp.ClassVar[str] = """
        @misc{romani2025brainform,
          doi = {10.48550/ARXIV.2510.10169},
          url = {https://arxiv.org/abs/2510.10169},
          author = {Romani, Michele and Zanoni, Devis and Farella, Elisabetta and Turchet, Luca},
          keywords = {Human-Computer Interaction (cs.HC), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, H.1.2; I.2.6; I.5.2},
          title = {BrainForm: a Serious Game for BCI Training and Data Collection},
          publisher = {arXiv},
          year = {2025},
          copyright = {arXiv.org perpetual, non-exclusive license}
        }

        @misc{romani2025dataset,
          author={Romani, Michele and Zanoni, Devis},
          title={The BrainForm Dataset},
          year={2025},
          publisher={Zenodo},
          doi={10.5281/zenodo.17225966},
          url={https://zenodo.org/records/17225966}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/17225966/files/BIDS.zip"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (8-ch) in 22 healthy participants while playing a BrainForm serious-game P300 speller."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=60,
        num_subjects=22,
        num_events_in_query=601,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 17127),
        frequency=250.0,
    )

    _BIDS_SUBDIR: tp.ClassVar[str] = "MNE-RomaniBF2025ERP-data/BIDS"
    _EXCLUDE_SUBJECTS: tp.ClassVar[set[str]] = {"P15", "P18"}

    def _bids_root(self) -> Path:
        return Path(self.path) / "download" / self._BIDS_SUBDIR

    def _subject_map(self) -> dict[int, str]:
        """Build {int_index: 'Pxx'} map consistent with MOABB's indexing."""
        bids = self._bids_root()
        labels = sorted(
            d.name.removeprefix("sub-")
            for d in bids.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        labels = [s for s in labels if s not in self._EXCLUDE_SUBJECTS]
        return {i: s for i, s in enumerate(labels)}

    def _download(self) -> None:
        """Build timelines.csv by scanning BIDS directory (no mne_bids needed)."""
        timeline_path = Path(self.path) / "timelines.csv"
        if timeline_path.exists():
            return

        bids = self._bids_root()
        if not bids.exists():
            raise FileNotFoundError(
                f"BIDS root not found at {bids}. Download the dataset first."
            )

        subj_map = self._subject_map()
        rows: list[dict[str, tp.Any]] = []
        for idx, label in subj_map.items():
            subj_dir = bids / f"sub-{label}"
            for ses_dir in sorted(subj_dir.iterdir()):
                if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                    continue
                ses_name = ses_dir.name.removeprefix("ses-")
                if "Failed" in ses_name:
                    continue
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.exists():
                    continue
                edf_files = list(eeg_dir.glob("*.edf"))
                if not edf_files:
                    continue
                rows.append(dict(subject=idx, session=ses_name, run="1calibration"))

        timelines = pd.DataFrame(rows)
        assert not timelines.empty, f"No timelines found from {bids}"
        timelines.to_csv(timeline_path, index=False)
        logger.info(
            f"Romani2025Brainform: wrote {len(timelines)} timelines to {timeline_path}"
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        """Load raw EDF + BIDS events.tsv directly, bypassing mne_bids lock files."""
        tl = timeline
        subj_map = self._subject_map()
        label = subj_map[tl["subject"]]
        ses = tl["session"]
        bids = self._bids_root()

        cache_key = (str(bids), tl["subject"], ses)
        if cache_key in _moabb_data_cache:
            _moabb_data_cache.move_to_end(cache_key)
            return _moabb_data_cache[cache_key]

        eeg_dir = bids / f"sub-{label}" / f"ses-{ses}" / "eeg"
        prefix = f"sub-{label}_ses-{ses}_task-ERP"
        edf_path = eeg_dir / f"{prefix}_eeg.edf"
        events_path = eeg_dir / f"{prefix}_events.tsv"
        channels_path = eeg_dir / f"{prefix}_channels.tsv"

        if not edf_path.exists():
            raise FileNotFoundError(f"EDF not found: {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # EDF doesn't carry channel-type metadata; MNE flags every channel as
        # ``eeg`` by default. The BrainForm recordings include a ``STI`` (TRIG)
        # hardware-trigger channel whose values directly encode the stimulus
        # codes (T1=1, T2=2, ..., T10=10). If left typed as ``eeg`` it leaks the
        # label into the model. Read ``*_channels.tsv`` and set MNE types so
        # that downstream ``picks=["eeg"]`` excludes non-brain channels.
        if channels_path.exists():
            ch_df = pd.read_csv(channels_path, sep="\t")
            type_map = {
                "EEG": "eeg",
                "EOG": "eog",
                "ECG": "ecg",
                "EMG": "emg",
                "TRIG": "stim",
                "RESP": "resp",
                "MISC": "misc",
                "REF": "misc",
            }
            ch_types = {
                row["name"]: type_map.get(str(row["type"]).upper(), "misc")
                for _, row in ch_df.iterrows()
                if row["name"] in raw.ch_names
            }
            if ch_types:
                raw.set_channel_types(ch_types, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        ev_df = pd.read_csv(events_path, sep="\t")
        n_targets = ev_df["value"].nunique()
        n_calib = 60 * n_targets
        if len(ev_df) > n_calib:
            ev_df = ev_df.iloc[:n_calib]

        descriptions = [
            "Target" if v == 1 else "NonTarget" for v in ev_df["value"].values
        ]
        raw.set_annotations(
            mne.Annotations(
                onset=ev_df["onset"].values,
                duration=ev_df["duration"].values,
                description=descriptions,
            )
        )

        if len(ev_df) >= n_calib:
            last_onset = ev_df["onset"].iloc[n_calib - 1]
            raw = raw.crop(tmin=0, tmax=last_onset + 1.0)

        _moabb_data_cache[cache_key] = raw
        while len(_moabb_data_cache) > _MOABB_CACHE_MAXSIZE:
            _moabb_data_cache.popitem(last=False)

        return raw


class Schaeff2012Exploring(_BaseMoabb):
    """Subset of MOABB: BNCI2015_007"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_007",)
    bibtex: tp.ClassVar[str] = """
        @article{schaeff2012exploring,
          title={Exploring motion VEPs for gaze-independent communication},
          volume={9},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2560/9/4/045006},
          DOI={10.1088/1741-2560/9/4/045006},
          number={4},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Schaeff, Sulamith and Treder, Matthias Sebastian and Venthur, Bastian and Blankertz, Benjamin},
          year={2012},
          month=jul,
          pages={045006}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 16 healthy participants while performing a gaze-independent motion-onset VEP (mVEP) P300 speller task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=32,
        num_subjects=16,
        num_events_in_query=2161,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 205132),
        frequency=100.0,
    )


# Suffixed to avoid collision with hand-written Schalk2004Bci2000 in schalk2004bci.py
class Schalk2004Bci2000Moabb(_BaseMoabb):
    """Subset of MOABB: PhysionetMI.

    MOABB wrapper for the motor imagery subset of the PhysioNet EEGMMIDB
    dataset. Only includes imagery runs with parsed trial labels
    (left_hand, right_hand, feet, hands, rest). See Schalk2004Bci2000 in
    schalk2004bci2000.py for the full dataset (all 14 runs per participant).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("PhysionetMI",)
    bibtex: tp.ClassVar[str] = """
        @article{schalk2004bci,
          title={BCI2000: A General-Purpose Brain-Computer Interface (BCI) System},
          volume={51},
          ISSN={0018-9294},
          url={http://dx.doi.org/10.1109/TBME.2004.827072},
          DOI={10.1109/tbme.2004.827072},
          number={6},
          journal={IEEE Transactions on Biomedical Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Schalk, G. and McFarland, D.J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J.R.},
          year={2004},
          month=jun,
          pages={1034-1043}
        }
    """
    url: tp.ClassVar[str] = "https://physionet.org/files/eegmmidb/1.0.0/"
    licence: tp.ClassVar[str] = "ODC-By-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 109 healthy participants while performing and imagining hand and feet movements for BCI cursor control."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "left_hand": 2,
        "right_hand": 3,
        "feet": 5,
        "hands": 4,
        "rest": 1,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=654,
        num_subjects=109,
        num_events_in_query=30,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 20000),
        frequency=160.0,
    )


class Scherer2015Individually(_BaseMoabb):
    """Subset of MOABB: BNCI2015_004"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_004",)
    bibtex: tp.ClassVar[str] = """
        @article{scherer2015individually,
          title={Individually Adapted Imagery Improves Brain-Computer Interface Performance in End-Users with Disability},
          volume={10},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0123727},
          DOI={10.1371/journal.pone.0123727},
          number={5},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Scherer, Reinhold and Faller, Josef and Friedrich, Elisabeth V. C. and Opisso, Eloy and Costa, Ursula and Kübler, Andrea and Müller-Putz, Gernot R.},
          editor={Bianchi, Luigi},
          year={2015},
          month=may,
          pages={e0123727}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (30-ch) in 9 healthy participants while performing five mental tasks (math, letter, rotation, count, baseline)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "math": 1,
        "letter": 2,
        "rotation": 3,
        "count": 4,
        "baseline": 5,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=18,
        num_subjects=9,
        num_events_in_query=201,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 714810),
        frequency=256.0,
    )


class Schirrmeister2017Deep(_BaseMoabb):
    """Subset of MOABB: Schirrmeister2017"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Schirrmeister2017",)
    bibtex: tp.ClassVar[str] = """
        @article{schirrmeister2017deep,
          title={Deep learning with convolutional neural networks for EEG decoding and visualization},
          volume={38},
          ISSN={1097-0193},
          url={http://dx.doi.org/10.1002/hbm.23730},
          DOI={10.1002/hbm.23730},
          number={11},
          journal={Human Brain Mapping},
          publisher={Wiley},
          author={Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
          year={2017},
          month=aug,
          pages={5391-5420}
        }
    """
    url: tp.ClassVar[str] = (
        "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (128-ch) in 14 healthy participants while performing executed movements (left hand, right hand, feet, rest)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "right_hand": 1,
        "left_hand": 2,
        "rest": 3,
        "feet": 4,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=28,
        num_subjects=14,
        num_events_in_query=321,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(128, 1226000),
        frequency=500.0,
    )


class Schreuder2010New(_BaseMoabb):
    """Subset of MOABB: BNCI2015_009"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_009",)
    bibtex: tp.ClassVar[str] = """
        @article{schreuder2010new,
          title={A New Auditory Multi-Class Brain-Computer Interface Paradigm: Spatial Hearing as an Informative Cue},
          volume={5},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0009813},
          DOI={10.1371/journal.pone.0009813},
          number={4},
          journal={PLoS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Schreuder, Martijn and Blankertz, Benjamin and Tangermann, Michael},
          editor={Yan, Jun},
          year={2010},
          month=apr,
          pages={e9813}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (20-ch) in 21 healthy participants while attending to spatial auditory cues in a multi-class AMUSE ERP paradigm."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=42,
        num_subjects=21,
        num_events_in_query=4321,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 452565),
        frequency=250.0,
    )


class Hohne2011Novel(_BaseMoabb):
    """Subset of MOABB: BNCI2015_012"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_012",)
    bibtex: tp.ClassVar[str] = """
        @article{hohne2011novel,
          title={A novel 9-class auditory ERP paradigm driving a predictive text entry system},
          volume={5},
          ISSN={1662-453X},
          url={http://dx.doi.org/10.3389/fnins.2011.00099},
          DOI={10.3389/fnins.2011.00099},
          journal={Frontiers in Neuroscience},
          publisher={Frontiers Media SA},
          author={Höhne, Johannes},
          year={2011}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (63-ch) in 10 healthy participants while attending to auditory stimuli varying in pitch and direction in a 9-class PASS2D P300 speller."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=20,
        num_subjects=10,
        num_events_in_query=2933,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 309720),
        frequency=250.0,
    )


class Schwarz2020Analyzing(_BaseMoabb):
    """Subset of MOABB: BNCI2020_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2020_001",)
    bibtex: tp.ClassVar[str] = """
        @article{schwarz2020analyzing,
          title={Analyzing and Decoding Natural Reach-and-Grasp Actions Using Gel, Water and Dry EEG Systems},
          volume={14},
          ISSN={1662-453X},
          url={http://dx.doi.org/10.3389/fnins.2020.00849},
          DOI={10.3389/fnins.2020.00849},
          journal={Frontiers in Neuroscience},
          publisher={Frontiers Media SA},
          author={Schwarz, Andreas and Escolano, Carlos and Montesano, Luis and Müller-Putz, Gernot R.},
          year={2020},
          month=aug
        }
    """
    url: tp.ClassVar[str] = "http://bnci-horizon-2020.eu/database/data-sets/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (58-ch) in 45 healthy participants while performing reach-and-grasp motor imagery comparing gel, water, and dry EEG electrode systems."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "palmar_grasp": 503587,
        "lateral_grasp": 503588,
        "rest": 768,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=45,
        num_subjects=45,
        num_events_in_query=162,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(58, 637184),
        frequency=256.0,
    )


class Shin2017OpenA(_BaseMoabb):
    """Subset of MOABB: Shin2017A"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Shin2017A",)
    bibtex: tp.ClassVar[str] = """
        @article{shin2017open,
          title={Open Access Dataset for EEG+NIRS Single-Trial Classification},
          volume={25},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/TNSRE.2016.2628057},
          DOI={10.1109/tnsre.2016.2628057},
          number={10},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Shin, Jaeyoung and von Luhmann, Alexander and Blankertz, Benjamin and Kim, Do-Won and Jeong, Jichai and Hwang, Han-Jeong and Muller, Klaus-Robert},
          year={2017},
          month=oct,
          pages={1735-1745}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/hBCI"
    licence: tp.ClassVar[str] = "GPL-3.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (30-ch) in 29 healthy participants while performing left- and right-hand motor imagery (opening/closing hands)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=87,
        num_subjects=29,
        num_events_in_query=21,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 120436),
        frequency=200.0,
    )


class Shin2017OpenB(_BaseMoabb):
    """Subset of MOABB: Shin2017B"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Shin2017B",)
    bibtex: tp.ClassVar[str] = """
        @article{shin2017open,
          title={Open Access Dataset for EEG+NIRS Single-Trial Classification},
          volume={25},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/TNSRE.2016.2628057},
          DOI={10.1109/tnsre.2016.2628057},
          number={10},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Shin, Jaeyoung and von Luhmann, Alexander and Blankertz, Benjamin and Kim, Do-Won and Jeong, Jichai and Hwang, Han-Jeong and Muller, Klaus-Robert},
          year={2017},
          month=oct,
          pages={1735-1745}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/hBCI"
    licence: tp.ClassVar[str] = "GPL-3.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (30-ch) in 29 healthy participants while performing mental arithmetic (serial subtraction) versus resting baseline."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"subtraction": 3, "rest": 4}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=87,
        num_subjects=29,
        num_events_in_query=21,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(30, 120236),
        frequency=200.0,
    )


class Sosulski2019Electroencephalogram(_BaseMoabb):
    """Subset of MOABB: Sosulski2019"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Sosulski2019",)
    bibtex: tp.ClassVar[str] = """
        @misc{sosulski2019electroencephalogram,
          doi = {10.6094/UNIFR/154576},
          url = {https://freidok.uni-freiburg.de/data/154576},
          author = {Sosulski, Jan and Tangermann, Michael},
          keywords = {612},
          language = {en},
          title = {Electroencephalogram signals recorded from 13 healthy subjects during an auditory oddball paradigm under different stimulus onset asynchrony conditions},
          publisher = {Albert-Ludwigs-Universität Freiburg},
          year = {2019}
        }
    """
    url: tp.ClassVar[str] = "https://freidok.uni-freiburg.de/dnb/download/154576"
    licence: tp.ClassVar[str] = "CC-BY-SA-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (31-ch) in 13 healthy participants while attending to target tones (1000 Hz) in an auditory oddball P300 paradigm."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 21, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=800,
        num_subjects=13,
        num_events_in_query=91,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(31, 29960),
        frequency=1000.0,
    )


class Srisrisawang2024Simultaneous(_BaseMoabb):
    """Subset of MOABB: BNCI2025_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2025_001",)
    bibtex: tp.ClassVar[str] = """
        @article{srisrisawang2024simultaneous,
          title={Simultaneous encoding of speed, distance, and direction in discrete reaching: an EEG study},
          volume={21},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2552/ada0ea},
          DOI={10.1088/1741-2552/ada0ea},
          number={6},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Srisrisawang, Nitikorn and Müller-Putz, Gernot R},
          year={2024},
          month=dec,
          pages={066042}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/001-2025/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (60-ch) in 20 healthy participants while performing discrete reaching movements in four directions at varying speeds and distances."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "up_slow_near": 1,
        "up_slow_far": 2,
        "up_fast_near": 3,
        "up_fast_far": 4,
        "down_slow_near": 5,
        "down_slow_far": 6,
        "down_fast_near": 7,
        "down_fast_far": 8,
        "left_slow_near": 9,
        "left_slow_far": 10,
        "left_fast_near": 11,
        "left_fast_far": 12,
        "right_slow_near": 13,
        "right_slow_far": 14,
        "right_fast_near": 15,
        "right_fast_far": 16,
    }
    # Non-brain channels (motion tracking, boolean validity flag, and target
    # positions from the reaching task) that need their type corrected to misc.
    _NON_EEG_CHANNELS: tp.ClassVar[dict[str, str]] = {
        "x": "misc",
        "y": "misc",
        "vx": "misc",
        "vy": "misc",
        "validity": "misc",
        "targetPosX": "misc",
        "targetPoxY": "misc",  # typo in original data (should be targetPosY)
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=20,
        num_subjects=20,
        num_events_in_query=959,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 4115020),
        frequency=500.0,
    )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raw = super()._load_raw(timeline)
        mapping = {
            ch: t for ch, t in self._NON_EEG_CHANNELS.items() if ch in raw.ch_names
        }
        if mapping:
            raw.set_channel_types(mapping)
        return raw


class Scherer2012Brain(_BaseMoabb):
    """Subset of MOABB: BNCI2014_002"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2014_002",)
    bibtex: tp.ClassVar[str] = """
        @article{scherer2012brain,
          title={Random forests in non-invasive sensorimotor rhythm brain-computer interfaces: a practical and convenient non-linear classifier},
          author={Steyrl, David and Scherer, Reinhold and Faller, Josef and Müller-Putz, Gernot R.},
          journal={Biomedical Engineering / Biomedizinische Technik},
          volume={61},
          number={1},
          pages={77--86},
          year={2016},
          doi={10.1515/bmt-2014-0117},
          url={https://doi.org/10.1515/bmt-2014-0117}
        }

        @article{scherer2012brainoriginal,
          title={Brain-computer interfacing: more than the sum of its parts},
          author={Scherer, Reinhold and Faller, Josef and Balderas, David and Friedrich, Elisabeth V. C. and Pröll, Markus and Allison, Brendan and Müller-Putz, Gernot},
          journal={Soft Computing},
          volume={17},
          number={2},
          pages={317--331},
          year={2012},
          doi={10.1007/s00500-012-0895-4},
          url={https://doi.org/10.1007/s00500-012-0895-4}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/"
    licence: tp.ClassVar[str] = "CC-BY-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 14 healthy participants while performing right-hand and feet motor imagery with auditory cues."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"right_hand": 1, "feet": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=112,
        num_subjects=14,
        num_events_in_query=21,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(15, 114176),
        frequency=512.0,
    )


class Stieger2021Continuous(_BaseMoabb):
    """Subset of MOABB: Stieger2021"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Stieger2021",)
    _dataset_kwargs: tp.ClassVar[dict[str, tp.Any]] = {"fix_bads": False}
    bibtex: tp.ClassVar[str] = """
        @article{stieger2021continuous,
          title={Continuous sensorimotor rhythm based brain computer interface learning in a large population},
          volume={8},
          ISSN={2052-4463},
          url={http://dx.doi.org/10.1038/s41597-021-00883-1},
          DOI={10.1038/s41597-021-00883-1},
          number={1},
          journal={Scientific Data},
          publisher={Springer Science and Business Media LLC},
          author={Stieger, James R. and Engel, Stephen A. and He, Bin},
          year={2021},
          month=apr
        }
    """
    url: tp.ClassVar[str] = "https://figshare.com/articles/dataset/13123148"
    licence: tp.ClassVar[str] = "CC-BY-NC-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 62 healthy participants while performing motor imagery for cursor control with continuous online visual feedback."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "right_hand": 1,
        "left_hand": 2,
        "both_hand": 3,
        "rest": 4,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=598,
        num_subjects=62,
        num_events_in_query=260,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 3876072),
        frequency=1000.0,
    )

    @classmethod
    def _get_dataset(cls) -> type[MoabbBaseDataset]:
        return find_dataset_in_moabb(cls.aliases[0], {"fix_bads": False})


class Tangermann2012Review(_BaseMoabb):
    """Subset of MOABB: BNCI2014_001"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2014_001",)
    bibtex: tp.ClassVar[str] = """
        @article{tangermann2012review,
          title={Review of the BCI Competition IV},
          volume={6},
          ISSN={1662-4548},
          url={http://dx.doi.org/10.3389/fnins.2012.00055},
          DOI={10.3389/fnins.2012.00055},
          journal={Frontiers in Neuroscience},
          publisher={Frontiers Media SA},
          author={Tangermann, Michael and Müller, Klaus-Robert and Aertsen, Ad and Birbaumer, Niels and Braun, Christoph and Brunner, Clemens and Leeb, Robert and Mehring, Carsten and Miller, Kai J. and Müller-Putz, Gernot R. and Nolte, Guido and Pfurtscheller, Gert and Preissl, Hubert and Schalk, Gerwin and Schlögl, Alois and Vidaurre, Carmen and Waldert, Stephan and Blankertz, Benjamin},
          year={2012}
        }
    """
    url: tp.ClassVar[str] = "https://lampx.tugraz.at/~bci/database/"
    licence: tp.ClassVar[str] = "CC-BY-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (22-ch) in 9 healthy participants while performing four-class motor imagery (left hand, right hand, feet, tongue)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "left_hand": 1,
        "right_hand": 2,
        "feet": 3,
        "tongue": 4,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=108,
        num_subjects=9,
        num_events_in_query=49,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(22, 96735),
        frequency=250.0,
    )


class Thielen2015Broad(_BaseCvepTrialMoabb):
    """Subset of MOABB: Thielen2015"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Thielen2015",)
    _code_length: tp.ClassVar[int] = 126
    _presentation_rate: tp.ClassVar[float] = 120.0
    _n_cycles: tp.ClassVar[int] = 4
    bibtex: tp.ClassVar[str] = """
        @article{thielen2015broad,
          title={Broad-Band Visually Evoked Potentials: Re(con)volution in Brain-Computer Interfacing},
          volume={10},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0133797},
          DOI={10.1371/journal.pone.0133797},
          number={7},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Thielen, Jordy and van den Broek, Philip and Farquhar, Jason and Desain, Peter},
          editor={Marinazzo, Daniele},
          year={2015},
          month=jul,
          pages={e0133797}
        }
    """
    url: tp.ClassVar[str] = "https://public.data.ru.nl/dcc/DSC_2018.00047_553_v3"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 12 healthy participants while performing a 6x6 matrix c-VEP BCI speller task using modulated Gold codes."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(36)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=36,
        num_subjects=12,
        num_events_in_query=145,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 542542),
        frequency=2048.0,
    )


class Thielen2021From(_BaseCvepTrialMoabb):
    """Subset of MOABB: Thielen2021"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Thielen2021",)
    _code_length: tp.ClassVar[int] = 126
    _presentation_rate: tp.ClassVar[float] = 60.0
    _n_cycles: tp.ClassVar[int] = 15
    bibtex: tp.ClassVar[str] = """
        @article{thielen2021from,
          title={From full calibration to zero training for a code-modulated visual evoked potentials brain computer interface},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2552/abecef},
          DOI={10.1088/1741-2552/abecef},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Thielen, Jordy and Marsman, Pieter and Farquhar, Jason and Desain, Peter},
          year={2021},
          month=mar
        }
    """
    url: tp.ClassVar[str] = "https://public.data.ru.nl/dcc/DSC_2018.00122_448_v3"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (8-ch) in 30 healthy participants while performing a code-modulated VEP (c-VEP) calculator BCI speller with pseudo-random sequences."
    )
    event_id: tp.ClassVar[dict[str, int]] = {f"symbol_{i}": i for i in range(20)}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=150,
        num_subjects=30,
        num_events_in_query=301,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(8, 341479),
        frequency=512.0,
    )


class Treder2011Gaze(_BaseMoabb):
    """Subset of MOABB: BNCI2015_008"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_008",)
    bibtex: tp.ClassVar[str] = """
        @article{treder2011gaze,
          title={Gaze-independent brain-computer interfaces based on covert attention and feature attention},
          volume={8},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2560/8/6/066003},
          DOI={10.1088/1741-2560/8/6/066003},
          number={6},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Treder, M S and Schmidt, N M and Blankertz, B},
          year={2011},
          month=oct,
          pages={066003}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 13 healthy participants while performing a gaze-independent center-speller P300 task with sequential geometric shapes."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=26,
        num_subjects=13,
        num_events_in_query=2041,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 777820),
        frequency=250.0,
    )


class Treder2014Decoding(_BaseMoabb):
    """Subset of MOABB: BNCI2015_006"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BNCI2015_006",)
    bibtex: tp.ClassVar[str] = """
        @article{treder2014decoding,
          title={Decoding auditory attention to instruments in polyphonic music using single-trial EEG classification},
          volume={11},
          ISSN={1741-2552},
          url={http://dx.doi.org/10.1088/1741-2560/11/2/026009},
          DOI={10.1088/1741-2560/11/2/026009},
          number={2},
          journal={Journal of Neural Engineering},
          publisher={IOP Publishing},
          author={Treder, M S and Purwins, H and Miklody, D and Sturm, I and Blankertz, B},
          year={2014},
          month=mar,
          pages={026009}
        }
    """
    url: tp.ClassVar[str] = "http://doc.ml.tu-berlin.de/bbci/"
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 11 healthy participants while selectively attending to individual instruments in polyphonic music in an auditory BCI task."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 1, "NonTarget": 2}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=11,
        num_subjects=11,
        num_events_in_query=1735,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 2277240),
        frequency=200.0,
    )


class VanVeen2019Building(_BaseMoabb):
    """Subset of MOABB: BI2012"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("BI2012",)
    bibtex: tp.ClassVar[str] = """
        @misc{vanveen2019building,
          doi = {10.5281/ZENODO.2649006},
          url = {https://zenodo.org/record/2649006},
          author = {Van Veen, Gijsbrecht Franciscus Petrus and Barachant, Alexandre and Andreev, Anton and Cattan, Grégoire and Coelho Rodrigues, Pedro Luis and Congedo, Marco},
          keywords = {Electroencephalography (EEG), P300, Brain-Computer Interface, Experiment},
          title = {Building Brain Invaders: EEG data of an experimental validation},
          publisher = {Zenodo},
          year = {2019},
          copyright = {Creative Commons Attribution 4.0 International}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/record/2649069/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (16-ch) in 25 healthy participants while playing a P300-based Brain Invaders video game."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"Target": 2, "NonTarget": 1}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=25,
        num_subjects=25,
        num_events_in_query=769,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(16, 46804),
        frequency=128.0,
    )


class Wang2017Benchmark(_BaseMoabb):
    """Subset of MOABB: Wang2016"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Wang2016",)
    bibtex: tp.ClassVar[str] = """
        @article{wang2017benchmark,
          title={A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces},
          volume={25},
          ISSN={1558-0210},
          url={http://dx.doi.org/10.1109/TNSRE.2016.2627556},
          DOI={10.1109/tnsre.2016.2627556},
          number={10},
          journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
          publisher={Institute of Electrical and Electronics Engineers (IEEE)},
          author={Wang, Yijun and Chen, Xiaogang and Gao, Xiaorong and Gao, Shangkai},
          year={2017},
          month=oct,
          pages={1746-1752}
        }

        @misc{wang2025benchmark,
          author={Wang, Yijun and Chen, Xiaogang and Gao, Xiaorong and Gao, Shangkai},
          title={A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces},
          year={2025},
          publisher={Zenodo},
          doi={10.5281/zenodo.14865172},
          url={https://zenodo.org/records/14865172}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/14865172/"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-ch) in 34 healthy participants while fixating on targets in a 40-class SSVEP speller using joint frequency-phase modulation."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "8": 1,
        "9": 2,
        "10": 3,
        "11": 4,
        "12": 5,
        "13": 6,
        "14": 7,
        "15": 8,
        "8.2": 9,
        "9.2": 10,
        "10.2": 11,
        "11.2": 12,
        "12.2": 13,
        "13.2": 14,
        "14.2": 15,
        "15.2": 16,
        "8.4": 17,
        "9.4": 18,
        "10.4": 19,
        "11.4": 20,
        "12.4": 21,
        "13.4": 22,
        "14.4": 23,
        "15.4": 24,
        "8.6": 25,
        "9.6": 26,
        "10.6": 27,
        "11.6": 28,
        "12.6": 29,
        "13.6": 30,
        "14.6": 31,
        "15.6": 32,
        "8.8": 33,
        "9.8": 34,
        "10.8": 35,
        "11.8": 36,
        "12.8": 37,
        "13.8": 38,
        "14.8": 39,
        "15.8": 40,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=34,
        num_subjects=34,
        num_events_in_query=241,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(64, 384000),
        frequency=250.0,
    )


class Wei2022BeetlA(_BaseMoabb):
    """Subset of MOABB: Beetl2021_A"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Beetl2021_A",)
    bibtex: tp.ClassVar[str] = """
        @inproceedings{wei2022beetl,
            title={2021 BEETL competition: Advancing transfer learning for subject independence and heterogenous EEG data sets},
            author={Wei, Xiaoxi and Faisal, A Aldo and Grosse-Wentrup, Moritz and Gramfort, Alexandre and Chevallier, Sylvain and Jayaram, Vinay and Jeunet, Camille and Bakas, Stylianos and Ludwig, Siegfried and Barmpas, Konstantinos and others},
            booktitle={NeurIPS 2021 Competitions and Demonstrations Track},
            pages={205--219},
            year={2022},
            organization={PMLR}
        }
    """
    url: tp.ClassVar[str] = "https://beetl.ai/data"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (63-ch) in 3 participants while performing motor imagery for the BEETL NeurIPS 2021 transfer-learning competition."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "rest": 0,
        "left_hand": 1,
        "right_hand": 2,
        "feet": 3,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=6,
        num_subjects=3,
        num_events_in_query=101,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 200000),
        frequency=500.0,
    )


class Wei2022BeetlB(_BaseMoabb):
    """Subset of MOABB: Beetl2021_B"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Beetl2021_B",)
    bibtex: tp.ClassVar[str] = """
        @inproceedings{wei2022beetl,
            title={2021 BEETL competition: Advancing transfer learning for subject independence and heterogenous EEG data sets},
            author={Wei, Xiaoxi and Faisal, A Aldo and Grosse-Wentrup, Moritz and Gramfort, Alexandre and Chevallier, Sylvain and Jayaram, Vinay and Jeunet, Camille and Bakas, Stylianos and Ludwig, Siegfried and Barmpas, Konstantinos and others},
            booktitle={NeurIPS 2021 Competitions and Demonstrations Track},
            pages={205--219},
            year={2022},
            organization={PMLR}
        }
    """
    url: tp.ClassVar[str] = "https://beetl.ai/data"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (32-ch) in 2 participants while performing motor imagery for the BEETL NeurIPS 2021 transfer-learning competition."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "rest": 3,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=4,
        num_subjects=2,
        num_events_in_query=121,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 96000),
        frequency=200.0,
    )


class Yi2014Evaluation(_BaseMoabb):
    """Subset of MOABB: Weibo2014"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Weibo2014",)
    bibtex: tp.ClassVar[str] = """
        @article{yi2014evaluation,
          title={Evaluation of EEG Oscillatory Patterns and Cognitive Process during Simple and Compound Limb Motor Imagery},
          volume={9},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0114853},
          DOI={10.1371/journal.pone.0114853},
          number={12},
          journal={PLoS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Yi, Weibo and Qiu, Shuang and Wang, Kun and Qi, Hongzhi and Zhang, Lixin and Zhou, Peng and He, Feng and Ming, Dong},
          editor={Maurits, Natasha M.},
          year={2014},
          month=dec,
          pages={e114853}
        }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (21-ch) in 10 healthy participants while performing simple (single-limb) and compound (multi-limb) motor imagery."
    )
    event_id: tp.ClassVar[dict[str, int]] = {
        "left_hand": 1,
        "right_hand": 2,
        "hands": 3,
        "feet": 4,
        "left_hand_right_foot": 5,
        "right_hand_left_foot": 6,
        "rest": 7,
    }
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=10,
        num_subjects=10,
        num_events_in_query=561,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 952000),
        frequency=200.0,
    )


class Zhou2016Fully(_BaseMoabb):
    """Subset of MOABB: Zhou2016"""

    aliases: tp.ClassVar[tuple[str, ...]] = ("Zhou2016",)
    bibtex: tp.ClassVar[str] = """
        @article{zhou2016fully,
          title={A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based Brain-Computer Interface},
          volume={11},
          ISSN={1932-6203},
          url={http://dx.doi.org/10.1371/journal.pone.0162657},
          DOI={10.1371/journal.pone.0162657},
          number={9},
          journal={PLOS ONE},
          publisher={Public Library of Science (PLoS)},
          author={Zhou, Bangyan and Wu, Xiaopei and Lv, Zhao and Zhang, Lei and Guo, Xiaojin},
          editor={He, Bin},
          year={2016},
          month=sep,
          pages={e0162657}
        }

        @misc{zhou2025bids,
          author={Zhou, Bangyan and Wu, Xiaopei and Lv, Zhao and Zhang, Lei and Guo, Xiaojin},
          title={Zhou2016 - BIDS},
          year={2025},
          publisher={Zenodo},
          doi={10.5281/zenodo.16534752},
          url={https://zenodo.org/records/16534752}
        }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/api/records/16534752"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (9-ch) in 4 healthy participants while performing three-class motor imagery (left hand, right hand, feet)."
    )
    event_id: tp.ClassVar[dict[str, int]] = {"left_hand": 1, "right_hand": 2, "feet": 3}
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=24,
        num_subjects=4,
        num_events_in_query=91,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(14, 305250),
        frequency=250.0,
    )
