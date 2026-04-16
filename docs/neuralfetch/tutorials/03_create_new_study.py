"""
Creating Your Own Study
=======================

Implement a custom Study subclass: define timelines, generate events,
and validate metadata so your dataset integrates with the full
neuralset + neuraltrain pipeline.

Continue from: :doc:`/neuralfetch/auto_examples/02_plot_study_anatomy`
"""

# %%
# Define a custom Study
# ----------------------

import typing as tp

import mne
import numpy as np
import pandas as pd

from neuralset.events import study as studies


class MyDemoStudy2026(studies.Study):
    _info: tp.ClassVar[studies.StudyInfo] = studies.StudyInfo(
        num_timelines=2,
        num_subjects=2,
        num_events_in_query=6,
        event_types_in_query={"Eeg", "Word"},
        data_shape=(8, 5000),
        frequency=100.0,
    )

    def _download(self) -> None:
        pass  # data is generated on the fly

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(2):
            yield {"subject": f"sub-{i:02d}"}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info_obj = studies.SpecialLoader(
            method=self._load_raw, timeline=timeline
        ).to_json()
        events = [
            dict(start=0, type="Eeg", filepath=info_obj),
            dict(start=1.0, duration=0.3, type="Word", text="hello"),
            dict(start=3.0, duration=0.3, type="Word", text="world"),
        ]
        return pd.DataFrame(events)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        n_chans, sfreq, duration = 8, 100.0, 50.0
        info = mne.create_info(n_chans, sfreq=sfreq, ch_types="eeg")
        data = np.random.RandomState(42).randn(n_chans, int(sfreq * duration)) * 1e-6
        return mne.io.RawArray(data, info, verbose=False)


# %%
# Load and inspect
# -----------------

import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
study = MyDemoStudy2026(path=tmp, infra_timelines={"cluster": None})
events = study.run()

print(f"Timelines: {events['timeline'].nunique()}")
print(f"Events: {len(events)}")
print(events[["type", "start", "duration", "text"]].head(10))

# %%
# Plot
# -----

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
for tl, df in events.groupby("timeline"):
    words = df[df["type"] == "Word"]
    ax.scatter(words["start"], [tl] * len(words), marker="|", s=200)
ax.set_xlabel("Time (s)")
ax.set_title("Word events per timeline")
plt.tight_layout()
plt.show()

# %%
# Cleanup
import shutil

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
