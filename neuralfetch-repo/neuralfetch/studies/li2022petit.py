# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import zipfile

import pandas as pd

from neuralfetch import download
from neuralset.events import study
from neuralset.events.etypes import Event


class Li2022Petit(study.Study):
    """Le Petit Prince (LPPC-fMRI): Multilingual fMRI corpus with ecological stimuli.

    This study provides a multilingual fMRI dataset where participants listened to
    "Le Petit Prince" (The Little Prince) audiobook in their native language
    (French, English, or Chinese). The same narrative content across languages
    enables cross-linguistic comparisons of natural language processing.

    Experimental Design:
        - 3T fMRI recordings (TR = 2.0 seconds)
        - 112 participants across three languages:
            * ~28 French native speakers (FR001-FR030, excluding FR021, FR027)
            * ~49 English native speakers (EN057-EN115, with some missing)
            * ~35 Chinese native speakers (CN001-CN037, excluding CN012, CN035)
        - 9 runs (sections) per participant
        - Paradigm: naturalistic listening (passive auditory comprehension)
            * 8-second delay (4 TRs) between trigger and stimulus onset
            * Each run is one section/chapter of the story
            * Stimulus language: French, English, or Chinese

    Data Format:
        - BIDS-compliant dataset structure (OpenNeuro ds003643)
        - Preprocessed BOLD data in MNIColin27 space
        - 1,008 total timelines (112 participants x 9 runs)
        - Event types: Fmri, Audio, Word, Text
        - TextGrid files: Word-level timing annotations
        - Transcript files: Full text of each section (password-protected)

    Download Requirements:
        - OpenNeuro dataset: ds003643 (version 2.0.5 or later)
        - openneuro-py, praatio
        - Transcript files must be manually obtained:
            * Contact neuralset maintainers for access
            * Files placed in: {path}/transcripts/lpp_{lang}_text.zip

    Notes:
        - Run numbers in filenames may differ from actual run order due to scanning
          issues or participant breaks.
        - Some participants removed in later dataset versions (FR021, FR027).
        - Multiple text fixes applied for alignment between audio and transcripts.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Le Petit Prince fMRI Corpus", "LPPC-fMRI")

    bibtex: tp.ClassVar[str] = """
    @article{li2022petit,
        title={Le Petit Prince multilingual naturalistic fMRI corpus},
        author={Li, Jixing and Bhattasali, Shohini and Zhang, Shulin and Franzluebbers, Berta and Luh, Wen-Ming and Spreng, R Nathan and Brennan, Jonathan R and Yang, Yiming and Pallier, Christophe and Hale, John},
        journal={Scientific data},
        volume={9},
        number={1},
        pages={530},
        year={2022},
        publisher={Nature Publishing Group UK London},
        doi={10.1038/s41597-022-01625-7},
        url={https://www.nature.com/articles/s41597-022-01625-7}
    }

    @misc{li2024petit,
        url = {https://openneuro.org/datasets/ds003643/versions/2.0.5/}
        author={Jixing Li and John Hale and Christophe Pallier},
        title={"Le Petit Prince: A multilingual fMRI corpus using ecological stimuli"},
        year={2024},
        doi={10.18112/openneuro.ds003643.v2.0.5},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds003643/versions/2.0.5/}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Le Petit Prince fMRI Corpus: 3T fMRI recordings from 112 participants "
        "listening to 'The Little Prince' audiobook in their native language "
        "(French, English, Chinese)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("openneuro-py", "praatio")
    dataset_id: tp.ClassVar[str] = "ds003643"
    TR_FMRI_S: tp.ClassVar[float] = 2.0

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1008,
        num_subjects=112,
        num_events_in_query=1757,
        event_types_in_query={"Fmri", "Audio", "Word", "Text"},
        data_shape=(73, 90, 74, 283),
        frequency=0.5,
        fmri_spaces=("custom",),  # MNIColin27
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.version = "v3"

    def _download(self) -> None:
        """
        Data is available here:
        https://openneuro.org/datasets/ds003643/versions/2.0.5/
        Make sure to get the last version of the dataset.
        """
        dataset_id = "ds003643"
        path = self.path
        if path.name.lower() != self.__class__.__name__.lower():
            path = path / self.__class__.__name__
        client = download.Openneuro(study=dataset_id, dset_dir=path)
        client.download()
        # TODO when available, automate download of transcripts files

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self.path.exists():
            raise ValueError(f"No folder {self.path}")
        lang_subjects = [
            ("FR", range(1, 31)),
            ("EN", range(57, 116)),
            ("CN", range(1, 38)),
        ]
        en_missing = [60, 66, 71, 80, 85, 90, 102, 107, 111, 112]
        for lang, subject_indices in lang_subjects:
            for sub_index in subject_indices:
                if lang == "FR" and sub_index in [21, 27]:
                    continue  # Those subjects have been removed in the last version
                if lang == "EN" and sub_index in en_missing:
                    continue  # does not exist
                if lang == "CN" and sub_index in [12, 35]:
                    continue  # does not exist
                for run in range(1, 10):
                    subject = f"sub-{lang}{sub_index:03d}"
                    yield dict(subject=subject, lang=lang, run=run)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load events"""
        from praatio import textgrid as ptg

        tl = timeline
        dl = self.path / "download"
        # word events
        txt_grid = (
            dl / f"annotation/{tl['lang']}/lpp{tl['lang']}_section{tl['run']}.TextGrid"
        )
        if not txt_grid.exists():
            raise FileNotFoundError(
                f"Missing TextGrid for {self.__class__.__name__}: {txt_grid}. "
                "Please run study.download() first."
            )
        tg = ptg.openTextgrid(str(txt_grid), includeEmptyIntervals=False)
        keys = tg.tierNames
        if len(keys) > 1:
            raise RuntimeError(f"Only one key should be in textgrid, got {keys}")
        # fixes to match text and annotations
        repl = {
            "three_hundred_twenty-five": "325",
            "six_hundred_twelve": "612",
            "one_thousand_nine_hundred_nine": "1909",
            "one_thousand_nine_hundred_twenty": "1920",
            "minster": "minister",
            'na\\i""ve': "naive",
            "coeur": "cœur",
            "oeil": "œil",
            "ll": "il",
            "Â": "",  # garbled encoding artifact in annotation
        }
        wordseq: list[dict[str, tp.Any]] = []
        duplicated = "it the this that i i. they we he she you now and but so there five twenty phew good".split()
        for interval in tg.getTier(keys[0]).entries:
            if interval.label.strip() in ("", "#", "sil"):
                continue
            # duplicated words happen a lot (when missing new sentence character), merge them
            if interval.label in duplicated and wordseq:
                if wordseq[-1]["text"] == interval.label:
                    wordseq[-1]["duration"] = interval.end - wordseq[-1]["start"]
                    continue
            # add word
            text = (
                repl.get(interval.label, interval.label)
                .replace("`", "'")
                .replace("«", "")
                .replace("»", "")
            )
            if not text:
                continue
            wordseq.append(
                {
                    "text": text,
                    "start": interval.start,
                    "duration": interval.end - interval.start,
                }
            )

        events = pd.DataFrame(wordseq)
        events["type"] = "Word"
        # Example sound file: task-lppFR_section_1.wav
        sep = "-" if tl["lang"] == "EN" else "_"
        soundfile = dl / f"stimuli/task-lpp{tl['lang']}_section{sep}{tl['run']}.wav"
        # TODO need to validate start time in some way, it may be incorrect:
        # "There was a trigger at the beginning of each section and a delay of 8 s (4 TRs)
        # between the trigger and onset of stimulus presentation for all three languages"
        event = {
            "type": "Audio",
            "start": 0,
            "filepath": str(soundfile),
            "timeline": "tmp",
        }
        event = Event.from_dict(event).to_dict()  # populates duration
        events2 = [event]
        lang_lower = tl["lang"].lower()
        textzip = self.path / "transcripts" / f"lpp_{lang_lower}_text.zip"
        if not textzip.exists():
            raise RuntimeError(
                f"Transcripts file must be manually dumped in {textzip}. "
                "To get them, contact directly the maintainers of neuralset."
            )
        with zipfile.ZipFile(textzip) as archive:
            text = archive.read(
                f"lpp_{lang_lower}_text/text_{lang_lower}_run{tl['run']}.txt",
                pwd=b"lessentielestinvisiblepourlesyeux",
            ).decode("utf8")
        # text fixes
        if lang_lower == "en":
            text = text.replace("did I have this sense", "did I have to have this sense")
            text = text.replace("my little price", "my little prince")
            asteroids = "3 2 5, 3 2 6, 3 2 7, 3 2 8, 3 2 9, and 3 3 0"
            if asteroids in text:
                replasteroids = asteroids
                for num, word in [
                    (0, "zero"),
                    (2, "two"),
                    (5, "five"),
                    (6, "six"),
                    (7, "seven"),
                    (8, "eight"),
                    (9, "nine"),
                ]:
                    replasteroids = replasteroids.replace(str(num), word)
                text = text.replace(asteroids, replasteroids)
            text = text.replace("street lamp", "streetlamp")
            text = text.replace("red faced", "redfaced")
        elif lang_lower == "fr":
            for old, new in [("’", "'"), (" pusse ", " puisse ")]:
                text = text.replace(old, new)
        language = {"en": "english", "fr": "french", "cn": "chinese"}[lang_lower]
        # end of text fixes
        events2.append({"type": "Text", "text": text, "language": language})
        # update text start/duration from sound start/duration
        events2[-1].update({x: events2[-2][x] for x in ["start", "duration"]})

        # add fmri event
        fmri_folder = dl / "derivatives" / tl["subject"] / "func"
        if not fmri_folder.exists():
            raise RuntimeError(f"Subject fmri folder does not exist: {fmri_folder}")
        files = sorted(fmri_folder.glob("*_space-MNIColin27_desc-preproc_bold.nii.gz"))
        if not files:
            raise RuntimeError(f"No fMRI files found in {fmri_folder}")
        # the run in the filename may be different ("scanning issues or participants needing a break")
        # so this filename would have been incorrect:
        # f"{tl['subject']}_task-lpp{tl['lang']}_run-{tl['run']:02d}_space-MNIColin27_desc-preproc_bold.nii.gz"
        # Use run index if available, otherwise use sequential index
        niifile = files[tl["run"] - 1] if len(files) >= tl["run"] else files[0]
        freq = 1.0 / self.TR_FMRI_S
        events2.append(dict(type="Fmri", start=0, filepath=str(niifile), frequency=freq))
        out = pd.concat([events, pd.DataFrame(events2)], ignore_index=True)

        out.loc[out.type.isin(["Word", "Sentence", "Text"]), "modality"] = "heard"

        return out.reset_index(drop=True)


# # # # # mini dataset # # # # #


class Li2022PetitSample(Li2022Petit):
    """Ultra-minimal Le Petit Prince fMRI corpus: single subject, single run.

    Subject EN057 (English native speaker), story section 1 (run 15 in filename)
    for fastest testing. Combines all event types: fMRI + Audio + Word + Text
    Data size: ~80 MB (vs. 140 GB for full dataset)
    Perfect for CI/CD and quick validation.

    Note: The Li2022Petit dataset uses different run numbering in filenames due to
    scanning issues. EN057's section 1 corresponds to run 15 in the files.
    """

    description: tp.ClassVar[str] = (
        "Ultra-minimal Le Petit Prince fMRI: Single subject (EN057), run 1 only. "
        "Lightweight (~80MB) for rapid CI/CD testing and validation."
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=1520,
        event_types_in_query={"Audio", "Fmri", "Text", "Word"},
        data_shape=(73, 90, 74, 282),
        frequency=0.5,
        fmri_spaces={"custom"},
    )

    # URL for the English transcript files (run 1 only)
    _TRANSCRIPTS_URL: tp.ClassVar[str] = (
        "https://belcorentin.github.io/pi-aiche-dee/lpp_en_text.zip"
    )

    def _download(self) -> None:
        """Download ultra-minimal dataset: EN057 section 1 only."""
        dataset_id = "ds003643"
        path = self.path
        if path.name.lower() != self.__class__.__name__.lower():
            path = path / self.__class__.__name__

        include_patterns = [
            "dataset_description.json",
            # English TextGrid file for section 1
            "annotation/EN/lppEN_section1.TextGrid",
            # English audio file for section 1
            "stimuli/task-lppEN_section-1.wav",
            # fMRI data for EN057 run 15 only (section 1)
            "derivatives/sub-EN057/func/*run-15*",
        ]

        client = download.Openneuro(
            study=dataset_id, dset_dir=path, include=include_patterns
        )
        client.download()

        # Download English transcripts from hosted location
        self._download_transcripts(path)

    def _download_transcripts(self, path: tp.Any) -> None:
        """Download English transcript files."""
        import requests

        transcripts_dir = path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        zip_path = transcripts_dir / "lpp_en_text.zip"

        if zip_path.exists():
            return  # Already downloaded

        try:
            resp = requests.get(self._TRANSCRIPTS_URL, timeout=60)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
        except Exception as e:
            # Create placeholder with instructions if download fails
            raise RuntimeError(
                f"Failed to download transcripts from {self._TRANSCRIPTS_URL}: {e}\n"
                f"Please manually place English text files in {transcripts_dir}"
            )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield only subject EN057, section 1."""
        if not self.path.exists():
            raise ValueError(f"No folder {self.path}")

        subject = "sub-EN057"
        lang = "EN"
        yield dict(subject=subject, lang=lang, run=1)
