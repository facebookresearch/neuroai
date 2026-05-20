# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Multi-modal synthetic study used by the docs Code Builder page.

Generates a single shared dataset on disk that exposes events for every
neuro modality (MEG / EEG / iEEG / EMG / fMRI / Spikes / fNIRS) and every
stimulus type (Word / Image / Audio / Video / classification ``Stimulus``)
the Code Builder exposes — so any (neuro × stim) combination yields a
runnable script.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

_DURATION_S = 60.0
_N_STIM = 20
_WORDS = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    "runs",
    "fast",
    "today",
    "across",
    "the",
    "open",
    "field",
    "before",
    "the",
    "sun",
    "sets",
)


def _make_meg_like(folder, fname, ch_type, n_chans, sfreq, rng):
    """Write a tiny .fif with the requested channel type, only if missing."""
    import mne

    fp = folder / fname
    if fp.exists():
        return
    info = mne.create_info(
        ch_names=[f"{ch_type.upper()}{i:03d}" for i in range(n_chans)],
        sfreq=sfreq,
        ch_types=ch_type,
    )
    n_samples = int(sfreq * _DURATION_S)
    data = rng.randn(n_chans, n_samples).astype("float32") * 1e-6
    mne.io.RawArray(data, info, verbose="ERROR").save(
        fp,
        overwrite=True,
        verbose="ERROR",
    )


def _make_fnirs(folder, fname, n_chans, sfreq, rng):
    """Write a minimal SNIRF v1.1 file with continuous-wave amplitude data.

    Hand-rolled with h5py (also used by `_make_spikes` for the synthetic
    NWB file) to avoid pulling in `pysnirf2` just for testing. h5py +
    imageio[ffmpeg] are declared in `neuralset[tutorials]` so the docs
    Code Builder install line picks them up automatically. Covers the
    smallest schema MNE's `read_raw_snirf` accepts: `formatVersion`, a
    `nirs` group with `metaDataTags`, `probe` (wavelengths + 2D
    source/detector positions), and one `data1` block with one
    `measurementList<i>` per channel.
    """
    import h5py

    fp = folder / fname
    if fp.exists():
        return
    n_samples = int(sfreq * _DURATION_S)
    # fNIRS raw amplitude is positive (light intensity), not a centred signal.
    data = rng.rand(n_samples, n_chans).astype("float64") * 0.1 + 0.5
    time = np.arange(n_samples, dtype="float64") / sfreq
    n_optodes = max(1, n_chans // 2)

    with h5py.File(fp, "w") as f:
        f.create_dataset("formatVersion", data=np.bytes_("1.1"))
        nirs = f.create_group("nirs")
        md = nirs.create_group("metaDataTags")
        for k, v in (
            ("SubjectID", "examplemultimodal"),
            ("MeasurementDate", "2026-01-01"),
            ("MeasurementTime", "00:00:00"),
            ("LengthUnit", "mm"),
            ("TimeUnit", "s"),
            ("FrequencyUnit", "Hz"),
        ):
            md.create_dataset(k, data=np.bytes_(v))
        probe = nirs.create_group("probe")
        probe.create_dataset(
            "wavelengths",
            data=np.array([760.0, 850.0], dtype="float64"),
        )
        probe.create_dataset(
            "sourcePos2D",
            data=np.column_stack([np.arange(n_optodes), np.zeros(n_optodes)]).astype(
                "float64"
            ),
        )
        probe.create_dataset(
            "detectorPos2D",
            data=np.column_stack([np.arange(n_optodes), np.ones(n_optodes)]).astype(
                "float64"
            ),
        )
        data_grp = nirs.create_group("data1")
        data_grp.create_dataset("dataTimeSeries", data=data)
        data_grp.create_dataset("time", data=time)
        for i in range(n_chans):
            ml = data_grp.create_group(f"measurementList{i + 1}")
            ml.create_dataset("sourceIndex", data=np.int32((i % n_optodes) + 1))
            ml.create_dataset("detectorIndex", data=np.int32((i % n_optodes) + 1))
            ml.create_dataset("wavelengthIndex", data=np.int32((i // n_optodes) + 1))
            ml.create_dataset("dataType", data=np.int32(1))  # 1 = CW amplitude
            ml.create_dataset("dataTypeIndex", data=np.int32(1))


def _make_videos(folder, n, rng):
    """Write a handful of tiny mp4 clips using imageio + ffmpeg."""
    import imageio.v3 as iio

    for i in range(n):
        fp = folder / f"vid_{i:03d}.mp4"
        if fp.exists():
            continue
        # 24 frames @ 8 fps = 3 s — enough for any video model's clip length.
        frames = rng.randint(0, 255, (24, 32, 32, 3), dtype=np.uint8)
        iio.imwrite(fp, frames, fps=8, codec="libx264")


def _make_fmri(folder, fname, rng):
    import nibabel as nib

    fp = folder / fname
    if fp.exists():
        return
    n_vox, n_t = 12, int(_DURATION_S / 2.0)  # TR = 2s
    img = nib.Nifti1Image(
        rng.randn(n_vox, n_vox, n_vox, n_t).astype("float32"),
        np.eye(4),
    )
    img.header.set_zooms((3.0, 3.0, 3.0, 2.0))
    img.to_filename(fp)


def _make_images(folder, n, rng):
    from PIL import Image

    for i in range(n):
        fp = folder / f"img_{i:03d}.png"
        if fp.exists():
            continue
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(fp)


def _make_audio(folder, n, rng):
    import soundfile as sf

    for i in range(n):
        fp = folder / f"aud_{i:03d}.wav"
        if fp.exists():
            continue
        sf.write(fp, (rng.randn(8_000).astype("float32") * 0.1), 16_000)


def _make_spikes(folder, fname, rng, n_units=8, rate_hz=5.0):
    """Write a tiny NWB-style HDF5 file with spike trains."""
    import h5py

    fp = folder / fname
    if fp.exists():
        return
    spike_times: list[float] = []
    spike_times_index: list[int] = []
    cumulative = 0
    for _ in range(n_units):
        n = max(1, int(rng.poisson(rate_hz * _DURATION_S)))
        ts = np.sort(rng.uniform(0.0, _DURATION_S, size=n)).astype("float64")
        spike_times.extend(ts.tolist())
        cumulative += n
        spike_times_index.append(cumulative)
    with h5py.File(fp, "w") as f:
        units = f.create_group("units")
        units.create_dataset("id", data=np.arange(n_units, dtype="int32"))
        units.create_dataset("spike_times", data=np.asarray(spike_times))
        units.create_dataset(
            "spike_times_index",
            data=np.asarray(spike_times_index, dtype="int64"),
        )


class ExampleMultiModal(study.Study):
    """Multi-modal synthetic study (2 subjects × 2 runs).

    Each timeline carries one minute of synthetic recordings for **every**
    neuro modality the Code Builder supports (MEG / EEG / iEEG / EMG / fMRI),
    plus 20 image, word, audio and ``Stimulus`` events. Used by
    ``docs/neuralset/code_builder.rst`` so any (neuro × stim) combination
    has data to chew on without an external download.
    """

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=4,
        num_subjects=2,
        num_events_in_query=4 * (7 + 5 * _N_STIM),
        event_types_in_query={
            "Meg",
            "Eeg",
            "Ieeg",
            "Emg",
            "Fmri",
            "Fnirs",
            "Spikes",
            "Image",
            "Word",
            "Audio",
            "Video",
            "Stimulus",
        },
        data_shape=(0,),
        frequency=100.0,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.cluster = None  # purely sequential
        # Events embed absolute filepaths under `self.path`. Study's cache
        # UID excludes `path` (it varies per machine), so a cache populated
        # under one path keeps returning stale filepaths after the user
        # moves the studies folder. The dataset is tiny (~2 MB) and cheap
        # to regenerate; skip the events cache so the demo stays robust.
        self.infra_timelines.folder = None

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for s in (1, 2):
            for r in (1, 2):
                yield {"subject": f"sub-{s:02d}", "run": r}

    def _download(self) -> None:
        for tl in self.iter_timelines():
            self._materialize(tl)

    def _materialize(self, timeline: dict[str, tp.Any]):
        folder = self.path / timeline["subject"] / f"run-{timeline['run']}"
        folder.mkdir(parents=True, exist_ok=True)
        seed = (int(timeline["subject"][-2:]) - 1) * 100 + timeline["run"]
        rng = np.random.RandomState(seed)
        _make_meg_like(folder, "raw-meg.fif", "mag", 19, 120.0, rng)
        _make_meg_like(folder, "raw-eeg.fif", "eeg", 32, 256.0, rng)
        _make_meg_like(folder, "raw-ieeg.fif", "seeg", 16, 512.0, rng)
        _make_meg_like(folder, "raw-emg.fif", "emg", 8, 256.0, rng)
        _make_fnirs(folder, "raw-fnirs.snirf", 12, 10.0, rng)
        _make_fmri(folder, "bold.nii.gz", rng)
        _make_spikes(folder, "spikes.h5", rng)
        _make_images(folder, _N_STIM, rng)
        _make_audio(folder, _N_STIM, rng)
        _make_videos(folder, _N_STIM, rng)
        return folder

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        folder = self._materialize(timeline)
        rows: list[dict[str, tp.Any]] = []
        # neuro recordings
        for typ, fname in (
            ("Meg", "raw-meg.fif"),
            ("Eeg", "raw-eeg.fif"),
            ("Ieeg", "raw-ieeg.fif"),
            ("Emg", "raw-emg.fif"),
            ("Fnirs", "raw-fnirs.snirf"),
        ):
            rows.append(
                {
                    "type": typ,
                    "filepath": str(folder / fname),
                    "start": 0.0,
                    "duration": _DURATION_S,
                }
            )
        rows.append(
            {
                "type": "Fmri",
                "filepath": str(folder / "bold.nii.gz"),
                "start": 0.0,
                "duration": _DURATION_S,
                "frequency": 0.5,
            }
        )
        rows.append(
            {
                "type": "Spikes",
                "filepath": str(folder / "spikes.h5"),
                "start": 0.0,
                "duration": _DURATION_S,
                "frequency": 200.0,
                "subject": str(timeline["subject"]),
            }
        )
        # stim events (uniformly spread across the recording)
        rng = np.random.RandomState(int(timeline["run"]))
        classes = ("face", "object", "scene")
        for i in range(_N_STIM):
            t = 1.0 + i * 1.0
            rows.append(
                {
                    "type": "Word",
                    "start": t,
                    "duration": 0.3,
                    "text": _WORDS[i % len(_WORDS)],
                    "language": "english",
                }
            )
            rows.append(
                {
                    "type": "Image",
                    "start": t,
                    "duration": 0.3,
                    "filepath": str(folder / f"img_{i:03d}.png"),
                }
            )
            rows.append(
                {
                    "type": "Audio",
                    "start": t,
                    "duration": 0.5,
                    "filepath": str(folder / f"aud_{i:03d}.wav"),
                }
            )
            rows.append(
                {
                    "type": "Video",
                    "start": t,
                    "duration": 1.0,
                    "filepath": str(folder / f"vid_{i:03d}.mp4"),
                }
            )
            rows.append(
                {
                    "type": "Stimulus",
                    "start": t,
                    "duration": 0.3,
                    "description": str(rng.choice(classes)),
                }
            )
        return pd.DataFrame(rows)
