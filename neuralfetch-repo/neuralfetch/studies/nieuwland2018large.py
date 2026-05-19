# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# type: ignore

import logging
import os
import typing as tp
import zipfile
from functools import lru_cache
from itertools import chain
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from pydantic import field_validator
from tqdm import tqdm

from neuralfetch import download, utils
from neuralset import utils as nutils
from neuralset.base import BaseModel
from neuralset.events import study

SITES = (
    "BIRM",
    "BRIS",
    "EDIN",
    "GLAS",
    "KENT",
    "LOND",
    "OXFO",
    "YORK",
)  # "STIR") # FIXME: STIR durations are wrong
logger = logging.getLogger(__name__)


class Nieuwland2018Large(study.Study):
    url = "https://osf.io/eyzaq/"
    """Nieuwland2018Large: EEG responses to predictable and unpredictable words in sentences.

    This dataset is a large-scale multi-site replication study investigating
    probabilistic prediction in language comprehension. Participants read sentences
    via rapid serial visual presentation (RSVP) while EEG was recorded. The study
    was conducted across 8 laboratories in the United Kingdom.

    Experimental Design:
        - EEG recordings (63-channel, 500 Hz)
        - 295 participants across 8 sites
        - Paradigm: reading sentences via RSVP
            * Delong replication items: article/noun prediction (expected vs. unexpected)
            * Control items: correct vs. incorrect sentence completions
            * Stimulus language: English

    Notes:
        - EEG systems and channel counts vary by site (BrainVision, BioSemi, Neuroscan).
          The data_shape reflects the representative channel count after processing.
        - The class description references 334 participants from 9 sites, but the
          script processes 295 participants from 8 sites (STIR site is excluded).
    """

    bibtex: tp.ClassVar[str] = """@article{nieuwland2018large,
        title={Large-scale replication study reveals a limit on probabilistic prediction in language comprehension},
        author={Nieuwland, Mante S and Politzer-Ahles, Stephen and Heyselaar, Evelien and Segaert, Katrien and Darley, Emily and Kazanina, Nina and Von Grebmer Zu Wolfsthurn, Sarah and Bartolozzi, Federica and Kogan, Vita and Ito, Aine and others},
        journal={ELife},
        volume={7},
        pages={e33468},
        year={2018},
        publisher={eLife Sciences Publications, Ltd},
        doi={10.7554/eLife.33468},
        url={https://elifesciences.org/articles/33468}
    }


    @misc{nieuwland2021replication,
        title={Replication Recipe Analysis plan},
        url={osf.io/eyzaq},
        publisher={OSF},
        author={Nieuwland, Mante S and Huettig, Falk and King, Jean-Rémi},
        year={2021},
        month={Jul}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings from 334 participants across 9 laboratories reading sentences in "
        "RSVP to test phonological predictions."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("openpyxl",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=335,
        num_subjects=295,
        num_events_in_query=2604,
        event_types_in_query={"Eeg", "Word", "Sentence"},
        data_shape=(63, 1148472),
        frequency=500,
    )

    def _download(self) -> None:
        with download.success_writer(self.path / "download_all") as already_done:
            if already_done:
                return
            download.Osf(study="eyzaq", dset_dir=self.path, folder="download").download()
            _preproc_stimuli(self.path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # The dataset is composed of EEG acquired at different sites.
        # Each site consists of different devices, and necessitate different
        # reading functions.
        subjects = []
        for site in SITES:
            site_cls = _get_site_functions(site, self.path)
            subjects.append(site_cls.iter())
        for recording in chain(*subjects):  # type: ignore
            yield dict(path=str(self.path), **recording.export())

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        site = _get_site_functions(timeline["site"], timeline["path"])
        kwargs = (
            dict(task=timeline["task"]) if timeline.get("task") is not None else dict()
        )
        raw = site._load_raw(subject=timeline["site_subject"], **kwargs)
        raw.pick_types(eeg=True, stim=True)
        if raw.get_montage() is None:
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        site = _get_site_functions(site=timeline["site"], path=timeline["path"])
        raw = self._load_raw(timeline)
        assert raw.times[0] == 0
        kwargs = dict()
        if timeline.get("fix_log") is not None:
            kwargs["fix_log"] = timeline["fix_log"]
        if timeline.get("task") is not None:
            kwargs["task"] = timeline["task"]
        try:
            annots = site._load_events(subject=timeline["site_subject"], **kwargs)  # type: ignore
        except NotImplementedError:
            # if no _load_events has been defined, we can
            # rely on raw.Annotations directly.
            annots = _read_bdf_events(raw)
        events = _parse_annots(timeline["path"], annots)

        # clean up to fit bm api
        events = events.loc[events["is_word"]].reset_index()
        mapping = dict(word_id="word_index")
        events = events.rename(columns=mapping)

        events["start"] = events.onset
        events["type"] = "Word"
        events["language"] = "english"
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_event = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    duration=raw.times[-1],
                    filepath=info,
                )
            ]
        )
        events = pd.concat([eeg_event, events], ignore_index=True)

        keep = [
            "type",
            "start",
            "duration",
            "text",
            "word_index",
            "sequence_id",
            "language",
            "filepath",
        ]
        events = events[keep]

        events = utils.add_sentences(events)
        events.loc[events.type.isin(["Word", "Sentence", "Text"]), "modality"] = "read"

        return events


def _get_site_functions(site, path: str):
    site_classes = dict(
        BIRM=_Birm,
        BRIS=_Bris,
        EDIN=_Edin,
        GLAS=_Glas,
        KENT=_Kent,
        LOND=_Lond,
        OXFO=_Oxfo,
        STIR=_Stir,
        YORK=_York,
    )
    cls = site_classes[site.upper()]
    return cls(path)


@lru_cache
def _preproc_stimuli(path: str):
    path_download = Path(path) / "download"
    path_preprocessed = Path(path) / "prepare"
    stim_dir = path_preprocessed / "stimuli"
    stim_dir.mkdir(exist_ok=True, parents=True)
    zip_file = path_download / "Stimuli" / "Stimuli.zip"
    xls_file = stim_dir / "replication_items.xlsx"

    if not xls_file.exists():
        with zipfile.ZipFile(str(zip_file), "r") as archive:
            with open(xls_file, "wb") as f:
                xls = archive.read("Stimuli/Sentence Materials/REPLICATION_ITEMS.xlsx")
                f.write(xls)

    delong = pd.read_excel(xls_file, sheet_name="Delong_Replication")
    control = pd.read_excel(xls_file, sheet_name="Control_experiment")

    def badchar(string):
        for k in " :,=.":
            string = string.replace(k, "_")
        while "__" in string:
            string = string.replace("__", "_")
        return string

    def fix_columns(df):
        columns = dict((k, badchar(k).lower()) for k in df.keys())
        return df.rename(columns=columns)

    control = fix_columns(control)
    delong = fix_columns(delong)

    return control, delong


def _parse_annots(path: str, annot: mne.Annotations) -> pd.DataFrame:
    """
    Quoting the authors:

    Delong lists:
    001-100 or 200: cloze markers. NB, there are lots of items
    with zero cloze, but because software programs don't always
    send a zero, I use 200, and we'll have to recode later.
    101-180: item markers
    201: a-expected
    202: an-expected
    203: a-unexpected
    204: an-unexpected
    205: noun-expected
    206: noun-unexpected
    207: sentence final word-expected
    208: sentence final word-unexpected
    255: any other word

    Control list:
    No cloze markers
    101-180: item markers (same as the Delong lists, as I am
    limited to 0-255; the CW in the items of the two experiments
    are not identical)
    210: correct
    211: incorrect
    255: any other word
    """

    # parse events
    events_ = list()
    for a in annot:
        onset = a["onset"]
        a = a["description"].split("/")[1]
        if a.lower().startswith("s"):
            stim = int(a[1:])
        else:
            stim = 0

        event = dict(onset=onset, stim=stim)
        if stim == 0:
            pass
        elif stim in (250, 251):
            event["is_question"] = True
        elif stim == 210:
            event["correct"] = True
            event["is_word"] = True
        elif stim == 211:
            event["correct"] = False
            event["is_word"] = True
        elif stim == 255:
            event["is_word"] = True
        elif stim in range(1, 101):
            event["sequence_stim"] = stim
        elif stim in range(101, 181):
            event["item"] = stim
        elif stim in range(201, 209):
            mapping = {
                201: dict(is_a=True, expected=True),
                202: dict(is_an=True, expected=True),
                203: dict(is_a=True, expected=False),
                204: dict(is_an=True, expected=False),
                205: dict(is_noun=True, expected=True),
                206: dict(is_noun=True, expected=False),
                207: dict(is_final=True, expected=True),
                208: dict(is_final=True, expected=False),
            }
            event["is_word"] = True
            event.update(mapping[stim])
        elif stim >= 212:  # in (212, 213, 251, 252):
            event["is_word"] = True
        else:
            pass  # print(f"Unknown trigger: {stim}")
        events_.append(event)

    # Enrich events
    events = pd.DataFrame(events_)
    if "correct" not in events.keys():
        events["correct"] = None
    if "expected" not in events.keys():
        events["expected"] = None

    keys = (
        "is_word",
        "is_question",
        "is_final",
        "is_start",
        "is_a",
        "is_an",
        "is_noun",
        "sequence_stim",
    )
    for k in keys:
        if k in events.keys():
            events[k] = events[k].fillna(False)
        else:
            events[k] = False

    # add inter stimulus interval
    word_or_q = events["is_word"] | events["is_question"]
    isi = events.loc[word_or_q, "onset"].diff()
    events.loc[isi.index, "isi"] = isi.values
    next_isi = events.loc[word_or_q, "onset"].diff(-1)
    events.loc[isi.index, "is_final"] = next_isi.values < -1.5
    events.loc[isi.index[:-1] + 1, "is_start"] = next_isi.values[:-1] < -1.5
    events.loc[events.index[events["is_question"]], "is_start"] = False

    # Find delong starts
    starts = np.r_[0, events.index[events["is_final"]] + 1]
    # starts = events.query('is_start').index
    stops = starts[1:] - 1

    # delong lists
    for sequence_id, (i, j) in enumerate(zip(starts, stops)):
        # ids
        events.loc[i:j, "sequence_id"] = sequence_id

        assert (j - i) < 50
        if np.array_equal(events.loc[i:j].stim.unique(), [251]):
            # FIXME: questions?
            continue

        if (j - i) < 3:
            logger.debug(
                f"sequence {sequence_id}: too short, skip.",
                events.loc[i:j].stim.values,
            )
            continue

        # find sequence item
        e = events.loc[i:j]
        item = e.loc[(e["stim"] >= 101) & (e["stim"] <= 180), "stim"].unique()
        cloze = e.loc[(e["stim"] >= 1) & (e["stim"] <= 100), "stim"].unique()
        if not (len(cloze)) and sequence_id < 10:
            logger.debug(f"sequence {sequence_id}: missing cloze marker, skip.")
            continue
        if not len(item):
            logger.debug(f"sequence {sequence_id}: missing item marker, skip.")
            continue
        if len(item) > 1:
            logger.debug(
                f"sequence {sequence_id}: found multiple item values: "
                f"{item}, take first one."
            )
        item = item[0] - 100
        events.loc[i:j, "sequence_item"] = item
        if len(cloze):
            events.loc[i:j, "sequence_cloze"] = cloze[0]

        # word
        idx = e.index[e["is_word"] & ~e["is_question"]]
        assert events.loc[idx[-1], "is_final"]
        events.loc[idx, "word_id"] = range(len(idx))

        # sequence type
        control = e["correct"].isin([True, False]).sum()
        delong = e["expected"].isin([True, False]).sum()
        assert control or delong
        assert not (control and delong)

        if control:
            correct = np.any(e.correct)  # noqa
            events.loc[i:j, "sequence_correct"] = correct
            events.loc[i:j, "sequence_type"] = "control"
            events.loc[i:j, "sequence_expected"] = None

            # match with STIMULI
            control_df, _ = _preproc_stimuli(path)
            match = control_df.loc[control_df["item_number"] == item]
            assert len(match) == 1
            match = match.iloc[0]
            sentence = match.correct if correct else match.incorrect
            words = sentence.split()

        else:
            expected = np.any(e.expected)
            events.loc[i:j, "sequence_correct"] = None
            events.loc[i:j, "sequence_type"] = "delong"
            expected = np.any(events.loc[i:j, "expected"])
            events.loc[i:j, "sequence_expected"] = expected

            # match with STIMULI
            _, delong_df = _preproc_stimuli(path)
            match = delong_df.loc[delong_df["item_number"] == item]
            assert len(match) == 1
            match = match.iloc[0]

            if expected:
                article = match["expected"]
                noun = match["expected_1"]  # FIXME
            else:
                article = match["unexpected"]
                noun = match["unexpected_1"]  # FIXME
            sentence = " ".join(
                [match.sentence_context, article, noun, match.sentence_ending]
            )
            words = sentence.split()

        word_idx = e.index[e["is_word"]]
        try:
            events.loc[word_idx, "text"] = words
            events.loc[word_idx, "sentence"] = " ".join(words)
        except (ValueError, AssertionError):
            msg = f"sequence {sequence_id}: mismatch {len(word_idx)} triggers"
            msg += f" but {len(words)} words: {str(words)}, skip."
            logger.debug(msg)
    events.is_word = events.text.fillna("").astype(str).apply(lambda w: len(w) > 0)
    events.loc[events.index[events["is_word"]], "duration"] = 0.200
    events = events[events.text != "nan"]  # FIXME: removing the control sentences for now

    return events


def _read_bdf_raw(subject: int, folder: Path, postfix="") -> mne.io.RawArray:
    precision = dict(edin=1, glas=3, lond=3, oxfo=1)
    site = folder.name.lower()
    assert folder.exists()
    fname = site + "{:0" + str(precision[site]) + "}" + postfix + ".bdf"
    fname = folder / fname.format(subject)
    misc = ["EXG%i" % i for i in range(1, 10)]
    misc += ["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"]
    raw = mne.io.read_raw_bdf(fname, eog=["VEOG", "HEOG"], misc=misc)
    assert raw.ch_names[-1] == "Status"  # stim channel
    return raw


def _read_bdf_events(raw: mne.io.RawArray) -> mne.Annotations:
    events = mne.find_events(raw, shortest_event=1)
    onsets = []
    descriptions = []
    for onset, _, stim in events:
        onsets.append(float(onset) / raw.info["sfreq"])
        descriptions.append("/S%i" % stim)
    return mne.Annotations(onsets, np.zeros(len(onsets)), descriptions)


class _Recording(BaseModel):
    """variables that define a recording"""

    site: str
    subject: str
    task: tp.Literal["main", "control"] | None = None
    fix_log: bool | None = None

    @field_validator("site", mode="before")
    def format_site(cls, value: tp.Any) -> str:
        site = value.__class__.__name__[1:].upper()
        if site not in SITES:
            raise ValueError(f"{site} is not in {SITES}")
        return site

    @field_validator("subject", mode="before")
    def format_subject(cls, value: int | str) -> str:
        return str(value)

    def export(self) -> dict[str, tp.Any]:
        out = self.dict()
        out["site_subject"] = self.subject
        out["subject"] = f"{self.site}_{self.subject}"
        return out


class BaseSite:
    """BaseClass for each recording to contain
    site-specific eeg, events and subject readers."""

    def __init__(self, path: str):
        self.site = self.__class__.__name__[1:].upper()
        self.path_download = (
            Path(path) / "download" / "Raw and Processed data" / "Raw data" / self.site
        )
        self.path_preprocessed = Path(path) / "prepare" / self.site

    def _load_events(
        self, subject: str, *, task: str | None = None, fix_log: bool | None = None
    ):
        raise NotImplementedError


class _Birm(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        subjects = range(1, 44)
        bad_subject = (7, 11, 14, 15, 16, 17, 37, 41, 42, 43)
        needs_log_fix = (9, 10, 12, 13, 14)
        for subject in subjects:
            if subject in bad_subject:
                continue
            # unzip files
            self._birm_prepare_files(subject)
            fix_log = subject in needs_log_fix
            yield _Recording(site=self, subject=subject, fix_log=fix_log)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        fname = self.path_preprocessed / f"birm{int(subject):02}.vhdr"
        raw = mne.io.read_raw_brainvision(fname, misc=["BIP1"], eog=["EOG"])
        return raw

    def _load_events(self, subject: str, *, fix_log: bool):  # type: ignore
        fname = self.path_preprocessed / f"birm{int(subject):02}.vmrk"
        annots = mne.read_annotations(fname, sfreq="auto", uint16_codec=None)

        # match log and eeg
        if fix_log:
            log_fname = self._list_logfiles(subject)
            log = self._parse_log(log_fname[0])
            out = self._match_log_eeg(log, annots)
        else:
            out = annots
        return out

    @staticmethod
    def _birm_clean_string(txt: str) -> str:
        """remove trailing and double spaces"""
        # remove double spaces
        txt = txt.strip()
        while "  " in txt:
            txt = txt.replace("  ", " ")
        return txt

    def _parse_log(self, fname: Path) -> pd.DataFrame:
        """Parse BIRM eprime file to retrieve triggers and sentences"""
        with open(str(fname), encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        txt_ = []
        for c in txt:
            encode = str(c.encode("ascii", errors="ignore"))[2:-1]
            if "\\n" in encode:
                continue
            elif "\\x07" in encode or "\\x01" in encode:
                txt_.append(" ")
            elif "\\x" not in encode:
                txt_.append(encode)
        txt = "".join(txt_)
        txt = txt.replace("\\t", "")
        txt = txt.replace("\\n", "")
        # txt = txt.replace('$', '')

        txt = self._birm_clean_string(txt)
        txt = txt.replace("Mr.", "Mr")
        txt = txt.replace("Mrs.", "Mrs")

        # split
        out = []
        trials = txt.split("Reading")[1:]
        trials[-1], _ = trials[-1].split("Data File")
        for trial in trials:
            # parse trigger
            raw_digits = "".join([c for c in trial if not c.isalpha()])

            digits = " ".join(
                [c.split()[0] for c in raw_digits.split("$") if c[0].isdigit()][:8]
            )

            trial_str = "".join([c for c in trial if not c.isdigit()])
            trial_str = self._birm_clean_string(trial_str)

            is_question = " Q " in trial_str
            if is_question:
                sent, question = trial_str.split(" Q ")
                question, sent2 = question.split("? ")
                sent2 = sent2.split(". ")[0]
                sent = sent + " " + sent2 + "."
                question += "?"
            else:
                assert " NQ " in trial_str
                sent, question = trial_str.split(" NQ ")
                if "." not in sent:
                    sent += " " + question.split(" .")[0]

            sent = sent.split(".")[0] + "."
            sent = sent.replace("  ", " ")
            assert sent[0].isupper()
            assert len(sent.split()) >= 9

            sent = self._birm_clean_string(sent)
            question = self._birm_clean_string(question)

            question_ = question if is_question else None
            out.append([sent, question_, digits, raw_digits])

        assert len(out) == 84

        df = pd.DataFrame(out, columns=["sequence", "question", "triggers", "raw"])

        def get_uid(x):
            x = x.replace("$", " ").split()
            x = map(int, x)
            x = sorted(set([i if i != 0 else 200 for i in x]))
            # for i in range(1, 10):
            #    if i in x:
            #        x.remove(i)
            return " ".join(map(str, x))

        df["sid"] = df.triggers.apply(get_uid).values
        return df

    @staticmethod
    def _match_log_eeg(log: pd.DataFrame, annots: mne.Annotations) -> mne.Annotations:
        # Parse annots to get sequence ids
        events_ = []
        for a in annots:
            onset = a["onset"]
            a = a["description"].split("/")[1]
            if a.lower().startswith("s"):
                stim = int(a[1:])
            else:
                stim = 0

            event = dict(onset=onset, stim=stim)
            events_.append(event)
        events = pd.DataFrame(events_)
        events["iti"] = events.onset.diff()  # type: ignore
        events["sequence_id"] = np.cumsum(events["iti"].values > 1.5)  # type: ignore  # noqa

        # match log log and eeg annots
        Y = log.sid
        X = []
        sids = []
        for sid, d in events.groupby("sequence_id"):
            x = " ".join(map(str, sorted(np.unique(d.stim.values))))
            if len(x.split()) < 3:
                continue
            X.append(x)
            sids.append(sid)

        # distance matrix
        D = np.zeros((len(X), len(Y)))
        # match matrix
        M = []  # type: ignore
        # knock out matrix
        K = np.ones((len(X), len(Y)))
        # prior matrix
        P = np.ones((len(X), len(Y)))

        pbar = tqdm(total=len(Y))
        while len(M) < len(Y):
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    if K[i, j]:
                        a, _ = nutils.match_list(x, y)
                        D[i, j] = len(a)

            prod = D * K * P
            if not prod.sum():
                break
            i, j = np.unravel_index(prod.argmax(), D.shape)  # type: ignore
            M.append([i, j])
            K[:, j] = 0
            K[i, :] = 0

            P[: len(Y)] += np.roll(np.eye(len(Y)), j - i) * 0.1
            P = np.clip(P, 0, 1.5)  # type: ignore
            pbar.n = len(M)
            pbar.update()
        pbar.close()

        # add fix stim
        events["valid_stim"] = False
        for i, j in M:
            d = events.loc[events["sequence_id"] == sids[i]]
            sid = log.loc[j].sid
            events.loc[d.index, "fix_sid"] = sid

            sid = np.array(list(map(int, sid.split())))
            for stim, d_ in d.groupby("stim"):
                if stim in sid:
                    events.loc[d_.index, "fix_stim"] = stim
                    events.loc[d_.index, "valid_stim"] = True
                elif stim in range(252, 255):
                    events.loc[d_.index, "fix_stim"] = 255
                    events.loc[d_.index, "valid_stim"] = True
                    events.loc[d_.index, "fix_by"] = 255 - stim
                else:
                    valid = False
                    for delta in range(-3, 4):
                        if stim in sid + delta:
                            events.loc[d_.index, "fix_stim"] = stim - delta
                            events.loc[d_.index, "fix_by"] = -delta
                            events.loc[d_.index, "valid_stim"] = True
                            valid = True
                            break
                    if not valid:
                        events.loc[d_.index, "fix_stim"] = stim

        # remove sequences that do not contain any valid stim
        valid_sid = []
        for sid, event_ in events.groupby("sequence_id"):
            if any(event_.valid_stim):
                valid_sid.append(sid)
        events = events.loc[events["sequence_id"].isin(valid_sid)]

        # make mne-like annotations
        annots = []
        for event_ in events.itertuples(index=False):
            if pd.isna(event_.fix_stim):
                stim = event_.stim
            else:
                stim = event_.fix_stim
            annots.append(dict(onset=event_.onset, description=f"/s{int(stim)}"))

        annots_ = mne.Annotations(
            onset=[d["onset"] for d in annots],
            duration=[0.0] * len(annots),  # FIXME
            description=[d["description"] for d in annots],
        )
        return annots_

    def _birm_prepare_files(self, subject):
        self.path_preprocessed.mkdir(exist_ok=True, parents=True)

        # prepare log files
        # check whether all files have already been extracted or linked
        exists = True
        for ext in ("vhdr", "vmrk", "eeg"):
            if not (self.path_preprocessed / (f"birm%.2i.{ext}" % subject)).exists():
                exists = False

        # prepare data
        if exists:
            return

        extras = (19, 20, 41, 42, 43)
        if subject in extras:
            archiv_fname = "Extra"
            exts = ("vhdr", "vmrk", "eeg")
        else:
            archiv_fname = "VHDR+VMRK"
            exts = ("vhdr", "vmrk")

        # extract zip file
        zip_file = self.path_download / f"{archiv_fname}.zip"
        with zipfile.ZipFile(str(zip_file), "r") as archive:
            for ext in exts:
                fname = f"birm{subject:02}.{ext}"
                with open(self.path_preprocessed / fname, "wb") as f:
                    f.write(archive.read(f"{archiv_fname}/{fname}"))

        # symbolic link for remaining files
        if subject not in extras:
            files = [
                f.name
                for f in (self.path_download).iterdir()
                if f"birm{subject:02}" in f.name.lower()
            ]
            for fname in files:
                assert (self.path_download / fname).exists()
                os.link(
                    str(self.path_download / fname), str(self.path_preprocessed / fname)
                )

    def _list_logfiles(self, subject):
        files = []
        for f in (self.path_preprocessed / "EPrime Output").iterdir():
            if f.suffix == ".edat2" and f"-{subject}-" in f.name:
                files.append(f)
        return files


class _Bris(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        bads = [32]  # missing file
        subjects = range(1, 40)
        for subject in subjects:
            if subject in bads:
                continue
            for task in ("main", "control"):
                yield _Recording(site=self, subject=subject, task=task)

    def _load_raw(
        self, subject: str, task: tp.Literal["main", "control"]
    ) -> mne.io.RawArray:
        fname = self.path_download / (f"bris{int(subject):02}_{task}.vhdr")
        raw = mne.io.read_raw_brainvision(str(fname))
        return raw

    def _load_events(
        self, subject: str, task: tp.Literal["main", "control"]
    ) -> mne.Annotations:
        fname = str(self.path_download / f"bris{int(subject):02}_{task}.vmrk")
        annots = mne.read_annotations(fname, sfreq="auto", uint16_codec=None)
        return annots


class _Edin(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        subjects = range(1, 36)
        for subject in subjects:
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        raw = _read_bdf_raw(int(subject), self.path_download)
        assert raw.ch_names[0] == "Fp1"
        assert raw.ch_names[63] == "O2"
        return raw


class _Glas(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        subjects = range(1, 36)
        for subject in subjects:
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        raw = _read_bdf_raw(int(subject), self.path_download)
        assert raw.ch_names[0] == "A1"
        assert raw.ch_names[127] == "D32"
        montage = mne.channels.make_standard_montage("biosemi128")
        raw.set_montage(montage)
        return raw


class _Kent(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        subjects = range(1, 39)
        for subject in subjects:
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        fname = str(self.path_download / f"kent{int(subject):04}.vhdr")
        raw = mne.io.read_raw_brainvision(fname, eog=["HEOG", "VEOG"])

        # montage
        assert raw.ch_names[0] == "Fp1"
        assert raw.ch_names[-4] == "HEOG"
        assert raw.ch_names[-3] == "VEOG"
        assert raw.ch_names[-2] == "A1"
        assert raw.ch_names[-1] == "A2"
        assert len(raw.ch_names) == 64

        mapping = dict((ch, ch.replace("Af", "AF")) for ch in raw.ch_names if "Af" in ch)
        mne.rename_channels(raw.info, mapping)
        return raw

    def _load_events(self, subject: str) -> mne.Annotations:
        fname = str(self.path_download / f"kent{int(subject):04}.vmrk")
        annots = mne.read_annotations(fname, sfreq="auto", uint16_codec=None)
        return annots


class _Lond(BaseSite):
    # bads = [3, 4]
    # subjects = [s for s in subjects if s not in bads]
    # ! two subjects have a different sfreq

    def _prepare(self, subject: int) -> None:
        raw_file = self.path_preprocessed / f"lond{subject:03}.bdf"
        # prepare
        if not raw_file.exists():
            if subject == 3:
                orig = "(answered questions randomly)"
            elif subject == 4:
                orig = "(possibly non native English speaker)"
            else:
                orig = ""

            source = self.path_download / f"lond{subject:03}{orig}.bdf"
            target = self.path_preprocessed / f"lond{subject:03}.bdf"
            os.link(str(source), str(target))
            if subject in (1, 2):
                source = self.path_download / f"lond{subject:03}_control.bdf"
                target = self.path_preprocessed / f"lond{subject:03}_control.bdf"
                os.link(str(source), str(target))

    def iter(self) -> tp.Iterator[_Recording]:
        # bads = [3, 4]
        subjects = range(1, 39)
        for subject in subjects:
            tasks = ("main", "control") if subject in (1, 2) else ("main",)
            for task in tasks:
                yield _Recording(site=self, subject=subject, task=task)

    def _load_raw(self, subject: str, task: str) -> mne.io.RawArray:
        self._prepare(int(subject))
        postfix = "" if task == "main" else "_control"
        raw = _read_bdf_raw(int(subject), self.path_preprocessed, postfix)
        assert raw.ch_names[0] == "Fp1"
        assert raw.ch_names[33] == "A2"
        assert raw.ch_names[34] == "VEOG"
        assert raw.ch_names[36] == "EXG5"
        return raw


class _Oxfo(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        for subject in range(1, 38):
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        raw = _read_bdf_raw(int(subject), self.path_download)
        assert raw.ch_names[0] == "Fp1"
        assert raw.ch_names[63] == "O2"
        return raw


class _Stir(BaseSite):
    def iter(self) -> tp.Iterator[_Recording]:
        for subject in range(1, 41):
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        # read raw
        fname = self.path_download / f"STIR{subject}.cnt"
        raw = mne.io.read_raw_cnt(
            fname, eog=("VEO", "HEO"), data_format="int32", misc=("CB1", "CB2")
        )

        for ch in raw.info["chs"]:
            ch["loc"][1] = ch["loc"][1] / 2.0
            ch["loc"][0] = ch["loc"][0] / 4.0
            ch["loc"][2] = np.clip(ch["loc"][2] - 0.5, 0, 1)
        # FIXME check montage?
        chs = ["FP1", "FP2", "FPZ", "FZ", "CZ", "PZ", "OZ"]
        mapping = dict((c, c.capitalize()) for c in chs)
        mapping["FCZ"] = "FCz"
        mapping["CPZ"] = "CPz"
        mapping["POZ"] = "POz"
        mne.rename_channels(raw.info, mapping)

        return raw

    def _load_events(self, subject: str) -> mne.Annotations:
        raw = self._load_raw(subject)
        annots = list(iter(raw.annotations))
        for annot in annots:
            stim = str(annot["description"])
            if stim.isdigit():
                stim = "/S" + stim
            else:
                stim = "/" + stim
            annot.update(dict(description=stim))
        return annots


class _York(BaseSite):
    # TODO zipfile REQUIREMENTS
    def _prepare(self, subject: str) -> None:
        vmrk_file = self.path_preprocessed / f"YORK{subject}.vmrk"
        vhdr_file = self.path_preprocessed / f"YORK{subject}.vhdr"
        eeg_file = self.path_preprocessed / f"YORK{subject}.eeg"

        # FIXME should have used utils.success_writer
        if any([not f.exists() for f in (vmrk_file, vhdr_file, eeg_file)]):
            # extract header
            if subject in range(1, 14):
                fname = "P1_to_P13_vmrk_vhdr.zip"
            elif subject in range(14, 39):
                fname = "P14_to_P38_vmrk_vhdr.zip"
            else:
                fname = "P39_to_P41_vmrk_vhdr.zip"

            zip_file = str(self.path_download / fname)
            with zipfile.ZipFile(zip_file, "r") as archive:
                with open(vmrk_file, "wb") as f:
                    f.write(archive.read(vmrk_file.name))
                with open(vhdr_file, "wb") as f:
                    f.write(archive.read(vhdr_file.name))
                os.link(self.path_download / eeg_file.name, eeg_file)

    def iter(self) -> tp.Iterator[_Recording]:
        for subject in range(1, 42):
            yield _Recording(site=self, subject=subject)

    def _load_raw(self, subject: str) -> mne.io.RawArray:
        vhdr_file = self.path_preprocessed / f"YORK{subject}.vhdr"
        raw = mne.io.read_raw_brainvision(vhdr_file, eog=["HEOG", "VEOG"], misc=["BIP1"])
        return raw

    def _load_events(self, subject: str) -> mne.Annotations:
        vmrk_file = self.path_preprocessed / f"YORK{subject}.vmrk"
        annots = mne.read_annotations(vmrk_file, sfreq="auto", uint16_codec=None)
        return annots
