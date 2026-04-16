# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import functools
import itertools
import os
import re
import typing as tp
import warnings
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

X = tp.TypeVar("X", covariant=True)


def all_subclasses(cls: tp.Type[X]) -> tp.Set[tp.Type[X]]:
    """Get all subclasses of cls recursively."""
    subs = set(cls.__subclasses__())
    return subs | {s for c in subs for s in all_subclasses(c)}


def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """
    from rapidfuzz.distance.Levenshtein import editops

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


ISSUED_WARNINGS = set()


def warn_once(message: str) -> None:
    if message not in ISSUED_WARNINGS:
        warnings.warn(message)
        ISSUED_WARNINGS.add(message)


# Define a dummy context manager to suppress output
@contextlib.contextmanager
def ignore_all() -> tp.Iterator[None]:
    with open(os.devnull, "w", encoding="utf8") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


class NoApproximateMatch(ValueError):
    """Error raised when the function could not fully match the two list
    The error has a 'match' attribute holding the matches so far, for debugging
    """

    def __init__(self, msg: str, matches: tp.Any) -> None:
        super().__init__(msg)
        self.matches = matches


@dataclasses.dataclass
class Tolerance:
    """Convenience tool for check if a value is  within tolerance"""

    abs_tol: float
    rel_tol: float

    def __call__(self, value1: float, value2: float) -> bool:
        diff = abs(value1 - value2)
        tol = max(self.abs_tol, self.rel_tol * min(abs(value1), abs(value2)))
        return diff <= tol


@dataclasses.dataclass
class Sequence:
    """Handle for current information on the sequence matching"""

    sequence: tp.Sequence[float]  # the sequence to match
    current: int  # the current index for next match look-up
    matches: list[int]  # the matches so far in the sequence

    def valid_index(self, shift: int = 0) -> bool:
        return self.current + shift < len(self.sequence)

    def diff(self, shift: int = 0) -> float:
        return self.sequence[self.current + shift] - self.last_value

    @property
    def last_value(self) -> float:
        return self.sequence[self.matches[-1]]

    def diff_to(self, ind: int) -> np.ndarray:
        r = self.matches[-1]
        sub = self.sequence[r : r + ind] if ind > 0 else self.sequence[r + ind : r]
        return np.array(sub) - self.last_value


def approx_match_samples(
    s1: tp.Sequence[float],
    s2: tp.Sequence[float],
    abs_tol: float,
    rel_tol: float = 0.003,
    max_missing: int = 3,
    first_match: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate sample sequence matching
    Eg:
    seq0 = [1100, 2300, 3600]
    seq1 = [0, 1110, 3620, 6500]
    will match on 1100-1110 with tolerance 10,
    and then on 3600-3620 (as the diffs match with tolerance 10)

    Returns
    -------
    tuple of indices which match on the first list and the second list
    """
    if first_match is None:
        # we need to figure out the first match:
        # let's try on for several initial matches,
        # and pick the one that matches the most
        success: tuple[np.ndarray, np.ndarray] | None = None
        error: tp.Any = None
        for offsets in itertools.product(range(max_missing + 1), repeat=2):
            try:
                out = approx_match_samples(
                    s1,
                    s2,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    max_missing=max_missing,
                    first_match=offsets,  # type: ignore
                )
                if success is None or len(out[0]) > len(success[0]):  # type: ignore
                    success = out
            except NoApproximateMatch as e:
                if error is None or error.matches[0][-1] < e.matches[0][-1]:
                    error = e
        if success is not None:
            return success
        raise error
    tolerance = Tolerance(abs_tol=abs_tol, rel_tol=rel_tol)
    seqs = (
        Sequence(s1, first_match[0] + 1, [first_match[0]]),
        Sequence(s2, first_match[1] + 1, [first_match[1]]),
    )
    while all(s.valid_index() for s in seqs):
        # if we match within the tolerance limit, then move on
        # otherwise move the pointer for the less advanced sequence
        if tolerance(seqs[0].diff(), seqs[1].diff()):
            for k, s in enumerate(seqs):
                s.matches.append(s.current)
                s.current += 1
        else:
            # move one step
            seqs[1 if seqs[1].diff() < seqs[0].diff() else 0].current += 1
        # allow for 1 extra (absolute) step if getting closer
        for k, seq in enumerate(seqs):
            other = seqs[(k + 1) % 2]
            if seq.valid_index(shift=1) and other.valid_index():
                # need to check 2 tolerance so that we can match farther
                # if it is closer
                if abs(seq.diff(1) - seq.diff()) <= 2 * abs_tol:
                    if abs(seq.diff(1) - other.diff()) < abs(seq.diff() - other.diff()):
                        seq.current += 1
        # if we are over the limit for matching, then abort
        if any(m.current - m.matches[-1] > max_missing + 1 for m in seqs):
            msg = f"Failed to match after indices {[s.matches[-1] for s in seqs]} "
            msg += f"(values {[s.last_value for s in seqs]}, {first_match=})\n"
            msg += f"(follows:\n {seqs[0].diff_to(10)}\n {seqs[1].diff_to(10)}"
            msg += f"(before:\n {seqs[0].diff_to(-10)}\n {seqs[1].diff_to(-10)}"
            out = tuple(np.array(s.matches) for s in seqs)  # type: ignore
            raise NoApproximateMatch(msg, matches=out)
    return tuple(np.array(s.matches) for s in seqs)  # type: ignore


def get_bids_filepath(
    root_path: str | Path,
    *,  # force to use keyword args to avoid bugs
    subject: int | str,
    run: int | str,
    session: int | str | None,
    task: str,
    filetype: tp.Literal["bold", "bold_mask", "bold_raw", "meg", "events"],
    data_type: tp.Literal["Fmri", "Meg"],
    space: str | None = None,
    ses_suffix: str = "",
    subj_suffix: str = "",
    run_suffix: str = "",
    ses_padding: str = "02",
    subj_padding: str = "02",
    run_padding: str = "02",
) -> Path:
    """Helper for loading BIDS format data"""
    root_path = Path(root_path)
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {root_path}")

    suffix = _get_bids_file_suffix(filetype, space)

    folder_dict = {"Fmri": "func", "Meg": "meg"}
    if session is not None:
        file_path = (
            root_path
            / f"sub-{subj_suffix}{int(subject):{subj_padding}}"
            / f"ses-{ses_suffix}{int(session):{ses_padding}}"  # type: ignore
            / folder_dict[data_type]
            / (
                f"sub-{subj_suffix}{int(subject):{subj_padding}}"
                f"_ses-{ses_suffix}{int(session):{ses_padding}}"
                f"_task-{task}"
                f"_run-{run_suffix}{int(run):{run_padding}}{suffix}"
            )
        )
    else:
        file_path = (
            root_path
            / f"sub-{subj_suffix}{int(subject):{subj_padding}}"
            / folder_dict[data_type]
            / (
                f"sub-{subj_suffix}{int(subject):{subj_padding}}"
                f"_task-{task}"
                f"_run-{run_suffix}{int(run):{run_padding}}{suffix}"
            )
        )
    return file_path


def get_bids_files(
    root_path: str | Path,
    task: str,
    filetype: tp.Literal["bold", "bold_mask", "bold_raw", "meg", "events"] | str,
    space: str | None = None,
    ses_suffix: str = "",
) -> tp.Iterator[dict[str, str]]:
    root_path = Path(root_path)

    suffix = _get_bids_file_suffix(filetype, space)

    for bids_file in root_path.glob(f"**/*task-{task}*{suffix}"):
        match = re.match(
            r"sub-(\d{2})_ses-"
            + f"{ses_suffix}"
            + r"(\d{2})"
            + f"_task-{task}"
            + r"_run-(\d{2})"
            + suffix,
            bids_file.name,
        )
        if match:
            yield dict(subject=match.group(1), session=match.group(2), run=match.group(3))
        else:
            raise ValueError(f"BIDS file {bids_file} does not match BIDS format")


def _get_bids_file_suffix(
    filetype: str,
    space: str | None = None,
) -> str:
    if filetype == "events":
        suffix = "_events.tsv"
    elif filetype == "bold_raw":
        suffix = "_bold.nii.gz"
    elif filetype == "bold":
        assert space is not None
        suffix = f"_space-{space}_desc-preproc_bold.nii.gz"
    elif filetype == "bold_mask":
        assert space is not None
        suffix = f"_space-{space}_desc-brain_mask.nii.gz"
    elif filetype == "meg":
        suffix = "_meg.fif"
    else:
        raise ValueError(
            f"filetype must be one of 'events', 'bold', 'bold_mask, 'bold_raw', or 'meg'... but is {filetype}"
        )
    return suffix


def read_bids_events(bids_events_df_fp: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        bids_events_df_fp, sep="\t", na_values="n/a", header="infer", **kwargs
    )


def get_masked_bold_image(
    bold_image: nibabel.Nifti1Image | nibabel.Nifti2Image,
    mask_image: nibabel.Nifti1Image | nibabel.Nifti2Image,
    output_nifti_version: int | None = None,
) -> nibabel.Nifti1Image | nibabel.Nifti2Image:
    mask_4d = np.expand_dims(mask_image.get_fdata().astype(bool), -1)
    masked_im_data = bold_image.get_fdata() * mask_4d
    if output_nifti_version is None:
        output_nifti_version = 2
        if isinstance(bold_image, nibabel.Nifti1Image):
            output_nifti_version = 1
    if output_nifti_version == 1:
        nifti_builder = nibabel.Nifti1Image
    elif output_nifti_version == 2:
        nifti_builder = nibabel.Nifti2Image
    else:
        msg = f"nifti_version must be either 1 or 2, got {output_nifti_version}"
        raise ValueError(msg)
    return nifti_builder(
        masked_im_data,
        affine=bold_image.affine,
        header=bold_image.header,
        file_map=bold_image.file_map,
    )


def get_spacy_model(*, model: str = "", language: str = "") -> tp.Any:
    """Returns a cached version of a spacy model based on either model name or language
    (defaults to using the language large version of the model)
    """
    if language and model:
        msg = f"Language and model cannot be specified at the same time, got {language=} and {model=}"
        raise ValueError(msg)
    if not language and not model:
        language = "english"  # default to English model
    if language:
        # Map ISO 639-1 codes to full language names
        _iso_to_lang = {
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "ja": "japanese",
            "zh": "chinese",
        }
        language = _iso_to_lang.get(language, language)
        defaults = dict(
            english="en_core_web_lg",
            french="fr_core_news_lg",
            spanish="es_core_news_lg",
            japanese="ja_core_news_lg",
            chinese="zh_core_web_lg",
        )
        if language not in defaults:
            raise ValueError(f"Language {language!r} not available: {defaults}")
        model = defaults[language]
    return _get_model(model)


@functools.lru_cache(maxsize=3)
def _get_model(model: str) -> tp.Any:
    import sys

    import spacy

    if not spacy.util.is_package(model):
        import shutil
        import subprocess

        if shutil.which("uv"):
            # uv-managed venvs have no pip; use uv with explicit --python so the
            # model lands in the current interpreter's venv, not the project default.
            # Replicate spacy's URL construction to bypass pip entirely.
            from spacy import about
            from spacy.cli.download import (
                get_compatibility,
                get_model_filename,
                get_version,
            )

            compatibility = get_compatibility()
            version = get_version(model, compatibility)
            filename = get_model_filename(model, version, sdist=False)
            url = about.__download_url__ + "/" + filename
            subprocess.check_call(
                ["uv", "pip", "install", "--python", sys.executable, url]
            )
        else:
            import spacy.cli

            spacy.cli.download(model)  # type: ignore
    nlp = spacy.load(model)
    return nlp


def train_test_split_by_group(
    *arrays: tp.Iterable[np.ndarray],
    groups: np.ndarray,
    test_size: float = 0.25,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Wrapper around GroupShuffleSplit to imitate train_test_split but with groups."""
    splitter = GroupShuffleSplit(
        test_size=test_size, n_splits=1, random_state=random_state
    )
    train_inds, test_inds = next(splitter.split(*arrays, groups=groups))

    splitting = []
    for arr in [*arrays, groups]:
        splitting.extend([arr[train_inds], arr[test_inds]])  # type: ignore

    return splitting
