"""
Create or share a study
=======================

Implement a custom Study subclass: define timelines, generate events,
and validate metadata so your dataset integrates with the full
neuralset + neuraltrain pipeline.
"""

# %%
# Define a custom Study
# ----------------------
#
# A minimal Study needs three methods: ``iter_timelines`` (which
# recording sessions exist?), ``_download`` (write the raw files to
# disk), and ``_load_timeline_events`` (the events for one session).

import typing as tp

import mne
import numpy as np
import pandas as pd

from neuralset.events import study as studies


class MyDemoStudy2026(studies.Study):
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # A timeline is one continuous recording (e.g. a subject's session).
        # Yield one dict of identifiers per timeline; the same dict is
        # passed back to ``_load_timeline_events`` below.
        for session in range(2):
            yield {"subject": f"sub-{session:02d}"}

    def _download(self) -> None:
        # Write the two raw .fif files neuralset will open below.
        info = mne.create_info(8, sfreq=100.0, ch_types="eeg")
        for subject in ("sub-00", "sub-01"):
            data = np.random.randn(8, 5000) * 1e-6
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.save(self.path / f"{subject}-raw.fif", overwrite=True)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # Return all events for a single timeline as a DataFrame. Each
        # row is one event; ``type`` distinguishes modalities (Eeg,
        # Word, Audio, Image, ...). Timing is in seconds.
        fif = self.path / f"{timeline['subject']}-raw.fif"
        return pd.DataFrame(
            [
                dict(start=0.0, type="Eeg", filepath=str(fif)),
                dict(start=1.0, duration=0.3, type="Word", text="hello"),
                dict(start=3.0, duration=0.3, type="Word", text="world"),
            ]
        )


# %%
# Load and inspect
# -----------------

import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
study = MyDemoStudy2026(path=tmp, infra_timelines={"cluster": None})
study.download()
events = study.run()

print(f"Timelines: {events['timeline'].nunique()}")
print(f"Events: {len(events)}")
print(events[["type", "start", "duration", "filepath", "text"]].head(10))

# %%
# Cleanup
import shutil

shutil.rmtree(tmp, ignore_errors=True)

# %%
# Advanced: declare metadata and stream neural data on demand
# ------------------------------------------------------------
#
# Production studies typically add two more pieces:
#
# 1. A class-level ``_info`` (a ``StudyInfo``) that records the expected
#    number of timelines, subjects, event counts, data shape and
#    sampling frequency. It powers automated tests that catch silent
#    regressions in loading code, and feeds the **Studies Explorer**
#    on the NeuralFetch landing page.
# 2. A ``SpecialLoader`` that defers neural-data loading. When a
#    recording is too large to keep on disk -- or is generated on the
#    fly -- skip the ``_download`` step and instead wrap a method bound
#    to a timeline in a ``SpecialLoader``, serialise it to JSON with
#    ``.to_json()``, and store the handle in the ``filepath`` column of
#    an ``Eeg`` / ``Meg`` / ``Audio`` event. Downstream transforms
#    (segmenter, extractors) deserialise the handle and call the method
#    only when they actually need the array.


class AdvancedDemoStudy2026(studies.Study):
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=2,
        num_subjects=2,
        num_events_in_query=3,
        event_types_in_query={"Eeg", "Word"},
        data_shape=(8, 5000),
        frequency=100.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in range(2):
            yield {"subject": f"sub-{session:02d}"}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # ``SpecialLoader`` packages a method + timeline into a JSON
        # handle. Stored on an ``Eeg`` event's ``filepath`` column, it
        # lets downstream transforms load the raw array lazily.
        eeg_handle = studies.SpecialLoader(
            method=self._load_raw, timeline=timeline
        ).to_json()
        return pd.DataFrame(
            [
                dict(start=0.0, type="Eeg", filepath=eeg_handle),
                # one row per event — (start, duration) in seconds, plus
                # any type-specific columns (here ``text`` for ``Word``).
                dict(start=1.0, duration=0.3, type="Word", text="hello"),
                dict(start=3.0, duration=0.3, type="Word", text="world"),
            ]
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        n_chans, sfreq, duration = 8, 100.0, 50.0
        info = mne.create_info(n_chans, sfreq=sfreq, ch_types="eeg")
        data = np.random.RandomState(42).randn(n_chans, int(sfreq * duration)) * 1e-6
        return mne.io.RawArray(data, info, verbose=False)


# %%
# Run the advanced study and inspect the events DataFrame -- note the
# ``filepath`` column on the ``Eeg`` rows now holds a serialised
# ``SpecialLoader`` handle.

tmp = Path(tempfile.mkdtemp())
advanced = AdvancedDemoStudy2026(path=tmp, infra_timelines={"cluster": None})
events = advanced.run()

print(f"Timelines: {events['timeline'].nunique()}")
print(f"Events: {len(events)}")
print(events[["type", "start", "duration", "filepath", "text"]].head(10))

shutil.rmtree(tmp, ignore_errors=True)

# %%
# Next steps
# -----------
#
# Your study is now registered and can be used anywhere a Study name is
# accepted — in ``ns.Study.catalog()``, in chains, and in training configs.
#
# See the **neuralset docs** to use your events DataFrame in a full
# training pipeline (extractors, segmenter, PyTorch DataLoader).
