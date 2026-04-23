# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

import neuralset as ns
from neuralset.base import Frequency
from neuralset.events import etypes, study
from neuralset.events.utils import extract_events


class Test2023Fmri(study.Study):
    # study/class level

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=3,
        num_events_in_query=11,
        event_types_in_query={"Fmri", "Word"},
        data_shape=(20, 20, 20, 10),  # 32-channel EEG
        frequency=0.5,
        fmri_spaces=("custom",),
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(3):
            yield {"subject": str(i)}

    def _load_raw(
        self, timeline: dict[str, tp.Any]
    ) -> nibabel.filebasedimages.FileBasedImage:
        # pylint: disable=unused-argument
        subject = timeline["subject"]
        nii = Path(self.path) / f"sub-{subject}.nii.gz"

        if not nii.exists():
            n_voxels = 20
            n_times = 10
            # lets vary number of voxels across subjects
            n_voxels += int(subject)
            data_array = np.random.rand(n_voxels, n_voxels, n_voxels, n_times)
            nifti_image = nibabel.Nifti1Image(data_array, np.eye(4))
            tr_in_seconds = 2.0
            nifti_image.header["pixdim"][4] = tr_in_seconds
            nii.parent.mkdir(exist_ok=True)
            nibabel.save(nifti_image, nii)
        return nibabel.load(nii)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sentences = "hello world. the quick brown fox. they quit. good bye."

        events = []
        start = 1.0
        splits = ["train", "test", "val"]
        for sid, sentence in enumerate(sentences.split(".")):
            if not sentence:
                continue
            sentence += "."
            for word in sentence.split():
                events.append(
                    dict(
                        start=start,
                        text=word,
                        duration=len(word) / 30,
                        type="Word",
                        language="english",
                        modality="heard",
                        split=splits[sid % 3],
                        sentence=sentence,
                    )
                )
                start += 0.5
            start += 2.0
        loader = study.SpecialLoader(method=self._load_raw, timeline=timeline)
        events.append(
            {
                "type": "Fmri",
                "filepath": loader.to_json(),
                "space": "custom",
                "start": 0,
                "frequency": 0.5,
                "duration": 20.0,
            }
        )
        return pd.DataFrame(events)


class Fake2025Fmri(study.Study):
    """Fake FMRI study with different areas lighting up for different events
    For Word and Image events, the area lights up differently for different instances.
    """

    # study/class level
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=2,
        num_events_in_query=22,
        event_types_in_query={"Fmri", "Image", "Audio", "Text", "Word"},
        data_shape=(91, 109, 91, 80),  # 32-channel EEG
        frequency=2,
        fmri_spaces=("custom",),
    )

    def _download(self) -> None:
        raise NotImplementedError

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(2):
            yield {"subject": "subj0", "run": i}
        yield {"subject": "subj1", "run": 0}

    # "timeline" is not used here but the uri serves for cache naming so must be unique
    # pylint: disable=unused-argument
    def _load_raw(
        self, timeline: dict[str, tp.Any]
    ) -> nibabel.filebasedimages.FileBasedImage:
        # NOTE: this method is not necessary if we can directly load a filepath
        # but we keep the URI pattern for testing purposes
        # and generate data on the fly to avoid dumping a useless big file
        df = self._load_timeline_events(timeline)
        df.loc[:, "timeline"] = "x"
        df.loc[:, "subject"] = timeline["subject"]
        events = extract_events(df)
        return generate_fmri(events)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        run = timeline["run"]
        duration = 40
        if duration < 12:
            raise RuntimeError("Need larger duration")
        path = Path(self.path)
        if path.name != self.__class__.__name__:
            path = path / self.__class__.__name__
        path.mkdir(exist_ok=True, parents=True)
        events: list[ns.events.Event] = []
        # FMRI
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        fmri = etypes.Fmri(
            filepath=info,
            subject=subject,
            start=0,
            frequency=2.0,
            timeline="x",
            duration=duration,
        )
        events.append(fmri)
        # fixed rng for data, run specific for timings
        seed = int(subject[4:]) * 573921 + run
        rng0 = np.random.RandomState(seed=12)
        rng = np.random.RandomState(seed=seed)
        # SOUND
        fp = path / f"fake-sound_{duration}s.wav"
        if not fp.exists():
            import scipy

            fps = 8_000
            audio = 0.002 * rng0.randn(int(duration - 10) * fps, 1)
            bip_sec = 0.3
            coeff = 2 * np.pi * 800  # 800 Hz
            tone = 0.5 * np.sin(coeff * np.linspace(0, bip_sec, int(fps * bip_sec)))
            start_i = int(0.5 * fps)
            while start_i < audio.shape[0] - len(tone):
                audio[start_i : start_i + len(tone), 0] += tone
                start_i += int(fps * min(5, max(0.5, start_i / fps * 0.25)))
                audio /= np.max(np.abs(audio))
            scipy.io.wavfile.write(fp, fps, audio)
        sound = etypes.Audio(filepath=fp, start=1 + run, timeline="x")
        events.append(sound)
        # WORDS
        sentences = [
            "Cat loves pineapple pizze.",
            "Dog hates pizze and pineapple.",
            "Cat hates dog pizze.",
        ]
        words = []
        start = rng.uniform(7)
        kwargs: tp.Any = {"timeline": "x", "language": "en"}
        for k, sentence in enumerate(sentences):
            parts = [s.strip(".,").lower() for s in sentence.split()]
            start += 2
            dur = 0.1
            for part in parts:
                w = etypes.Word(text=part, start=start, duration=dur, **kwargs)
                words.append(w)
                start += (k + 1) * 0.2
        start = min(e.start for e in words)
        dur = max(e.start + e.duration for e in words)
        text = etypes.Text(text=" ".join(sentences), start=start, duration=dur, **kwargs)
        events.extend(words + [text])
        # IMAGES
        filepaths = generate_images(4, path)
        start = 2
        while start < duration - 8:
            dur = rng.uniform(0.3, 1.5)
            fp = rng.choice(filepaths)  # type: ignore
            im = etypes.Image(start=start, duration=dur, filepath=fp, timeline="x")
            events.append(im)
            start += dur + rng.uniform(0, 5)
        out = pd.DataFrame([e.to_dict() for e in events])
        return out.drop(columns=["timeline", "subject"])


def generate_images(num: int, folder: Path) -> list[Path]:
    """generate images of concentric sinusoids"""
    rng0 = np.random.RandomState(seed=12)
    paths = []
    for k in range(num):
        paths.append(folder / f"{k}.png")
        if not paths[-1].exists():
            import PIL.Image

            array = 0.1 * rng0.normal(size=(128, 128, 3))
            c = rng0.randint(3)
            period = rng0.randint(32, 64)
            wave = (np.arange(128)[:, None] - rng0.randint(32, 96)) ** 2
            wave = wave + (np.arange(128)[None, :] - rng0.randint(32, 96)) ** 2
            wave = np.sin(2 * np.pi * np.sqrt(wave) / period)
            array[:, :, c] += wave
            array = (255 / 2 * (array + 1)).clip(0, 255)
            im = PIL.Image.fromarray(array.astype(np.uint8))
            im.save(paths[-1])
    return paths


def generate_fmri(events) -> tp.Any:
    from nilearn import datasets, image

    fmris = [e for e in events if isinstance(e, etypes.Fmri)]
    if len(fmris) != 1:
        raise RuntimeError("Only 1 FMRI generation is supported")
    duration = fmris[0].duration
    freq = Frequency(fmris[0].frequency)
    delay = 4.7  # some delay between stimuli and responses

    # AAL (MNI space)
    atlas = datasets.fetch_atlas_aal()
    atlas_img = image.load_img(atlas["maps"])
    atlas_inds = atlas_img.get_fdata()
    # small noise over the data
    data = 0.02 * np.random.uniform(size=atlas_inds.shape + (freq.to_ind(duration),))
    data[atlas_inds == 0, :] = 0  # background to 0
    label_ind = dict(zip(atlas.lut.name, atlas.lut["index"]))
    type_labels = {
        "Word": ["Frontal_Sup_Medial_L", "Frontal_Med_Orb_R"],
        "Audio": ["Temporal_Sup_L", "Temporal_Mid_R"],
        "Image": ["Lingual_L", "Lingual_R"],
    }
    record = {}
    # words and images
    rng = np.random.RandomState(len(events))
    for event in events:
        key = None
        if isinstance(event, etypes.Word):
            key = (event.type, event.text)
        elif isinstance(event, (etypes.Image, etypes.Audio)):
            key = (event.type, str(event.filepath))
        elif isinstance(event, (etypes.Text, etypes.Fmri)):
            continue  # nothing to do
        else:
            raise NotImplementedError(f"Unsupported event {event!r}")
        # create a mask for this type
        mask: np.ndarray | None = None
        for label in type_labels[event.type]:
            tmp = atlas_inds == label_ind[label]
            assert tmp.sum(), f"Missing data for label {label!r}"
            if not tmp.any():
                raise RuntimeError(f"Something went wrong for label {label!r}")
            if mask is None:
                mask = tmp
            else:
                mask |= tmp
        if mask is None:
            raise RuntimeError(f"Something went wrong for event {event!r} (no mask)")
        if not mask.any():
            raise RuntimeError(f"Something went wrong for type {event.type!r}")
        # generate the data for the event
        if key not in record:
            record[key] = rng.uniform(size=mask.sum())
        start = freq.to_ind(event.start + delay)
        if isinstance(event, etypes.Audio):
            import scipy.ndimage

            sound = event.read()  # (num_samples, num_channels)
            sound = abs(sound).sum(axis=1)
            ratio = event.frequency / freq
            if ratio != int(ratio):
                msg = "For simplicity's sake, sound frequencu should be a multiple of fMRI's"
                raise RuntimeError(msg)
            sound = scipy.ndimage.gaussian_filter1d(sound, sigma=2 * ratio)[:: int(ratio)]
            # normalize, but not fully just to have different dynamics
            sound /= max(sound) * 2
            data[mask, start : start + len(sound)] = record[key][:, None] * sound[None, :]
        else:
            dt = max(1, freq.to_ind(event.duration))
            data[mask, start : start + dt] += record[key][:, None]
    # return as nifiti
    nifti_image = nibabel.Nifti2Image(data, affine=atlas_img.affine)
    nifti_image.header["pixdim"][4] = 1 / freq
    return nifti_image
