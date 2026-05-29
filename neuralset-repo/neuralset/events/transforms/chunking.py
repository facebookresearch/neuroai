# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pandas as pd

from .. import etypes as ev
from ..study import EventsTransform

Tiling = tp.Literal["equal", "max"]


class _Section(tp.NamedTuple):
    """Contiguous ``[start, stop)`` span, label-homogeneous by construction."""

    start: float
    stop: float

    @property
    def duration(self) -> float:
        return self.stop - self.start

    def tile(
        self, min_duration: float, max_duration: float, tiling: Tiling
    ) -> list["_Section"]:
        """Partition into sub-sections per ``tiling`` strategy.

        ``"equal"``: maximally equal-sized chunks, each in ``[min_duration, max_duration]``.
        Raises if ``2 * min_duration > max_duration`` (bounds unsatisfiable) or
        if the section is shorter than ``min_duration``.

        ``"max"``: chunks of exactly ``max_duration`` plus a trailing partial
        covering the remainder. ``min_duration`` is unused here.
        """
        tol = ChunkEvents._SAMPLE_GRID_TOL
        if tiling == "equal":
            if 2 * min_duration > max_duration + tol:
                raise ValueError(
                    f"min_duration={min_duration} must be <= "
                    f"max_duration/2={max_duration / 2} so uniform tiling "
                    f"always yields chunks in [min_duration, max_duration]"
                )
            if self.duration < min_duration - tol:
                raise ValueError(
                    f"section [{self.start:.3f}, {self.stop:.3f}) (duration "
                    f"{self.duration:.3f}s) cannot tile into "
                    f"[{min_duration}, {max_duration}]. Set ``min_duration`` "
                    f"below the shortest section, merge short same-split runs "
                    f"in ``event_type_to_split_by`` before chunking, or use "
                    f'``tiling="max"`` to drop short trailing pieces instead.'
                )
            n = max(1, int(np.ceil(self.duration / max_duration - tol)))
            dur = self.duration / n
            return [
                _Section(self.start + i * dur, self.start + (i + 1) * dur)
                for i in range(n)
            ]
        # tiling == "max"
        if not np.isfinite(max_duration):
            return [self]  # post-filter handles ``duration < min_duration``
        n_full = int(self.duration / max_duration + tol)
        out = [
            _Section(self.start + i * max_duration, self.start + (i + 1) * max_duration)
            for i in range(n_full)
        ]
        if self.duration - n_full * max_duration > tol:
            out.append(_Section(self.start + n_full * max_duration, self.stop))
        return out


def _build_sections(
    event: ev.BaseSplittableEvent,
    use_rows: pd.DataFrame | None = None,
    allow_sample_leakage: bool = False,
) -> list[_Section]:
    """Partition ``event`` into label-homogeneous sections.

    Boundaries sit at the midpoint of the silence gap between consecutive
    rows of different ``split``. First section absorbs any leading silence;
    last extends to ``event.start + event.duration``. Returns a single
    section covering the whole event when ``use_rows`` is None/empty.

    Raises if a split transition leaves less than ``1 / frequency`` of
    silence (unless ``allow_sample_leakage=True``): ``event._split``'s
    nearest-grid snap of the midpoint isn't guaranteed to stay in the gap
    below that, so a labeled sample could end up in the wrong chunk.
    """
    tol = ChunkEvents._SAMPLE_GRID_TOL
    e_start, e_stop = event.start, event.start + event.duration
    if use_rows is None or len(use_rows) == 0:
        return [_Section(e_start, e_stop)]
    mask = (use_rows.start + use_rows.duration > e_start + tol) & (
        use_rows.start < e_stop - tol
    )
    rows = use_rows[mask].sort_values("start").reset_index(drop=True)
    if rows.empty:
        return [_Section(e_start, e_stop)]
    freq = float(event.frequency)
    min_gap = 1.0 / freq if freq > 0 else 0.0
    splits = rows.split.astype(str)
    boundaries = [e_start]
    for i in range(1, len(rows)):
        if splits.iloc[i] == splits.iloc[i - 1]:
            continue
        prev_stop = rows.iloc[i - 1].start + rows.iloc[i - 1].duration
        next_start = rows.iloc[i].start
        if not allow_sample_leakage and next_start - prev_stop < min_gap - tol:
            raise ValueError(
                f"label leakage: split transition at {prev_stop:.3f}s "
                f"({splits.iloc[i - 1]!r} → {splits.iloc[i]!r}) must leave "
                f"at least 1/frequency = {min_gap:.3f}s of silence at {freq} Hz. "
                f"Pass ``allow_sample_leakage=True`` to accept up to 1 sample "
                f"of mislabeling at the boundary."
            )
        boundaries.append((prev_stop + next_start) / 2)
    boundaries.append(e_stop)
    return [
        _Section(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
    ]


class ChunkEvents(EventsTransform):
    """Chunk long events into shorter events.

    Typical use: keep long recordings under a deep-learning model's memory
    budget (e.g. Wav2Vec).

    Parameters
    ----------
    event_type_to_chunk : str
        Splittable event type to chunk. Any
        :class:`~neuralset.events.etypes.BaseSplittableEvent` subclass
        (Audio, Video, Meg, Eeg, Fmri, ...).
    max_duration : float, default=``np.inf``
        Upper bound on chunk duration in seconds.
    min_duration : float, default=0.0
        Lower bound on chunk duration. Behavior when impossible depends on
        ``tiling`` (see below).
    tiling : {"max", "equal"}, default ``"max"``
        How each section is sub-divided:

        - ``"max"``: emit chunks of exactly ``max_duration`` until the section
          is exhausted; the trailing partial chunk is dropped iff its
          duration is ``< min_duration``.
        - ``"equal"``: equal-sized chunks, each in ``[min_duration, max_duration]``.
          Requires ``2 * min_duration <= max_duration``. Raises if a section
          is shorter than ``min_duration``.
    event_type_to_split_by : str, optional
        Align chunk boundaries with train/val/test labels carried by another
        event type's ``split`` column, to avoid label leakage at split
        transitions. When set, chunk boundaries follow same-``split`` runs
        and each run is sub-tiled per ``tiling``.
    allow_sample_leakage : bool, default=False
        Only relevant when ``event_type_to_split_by`` is set. If True,
        accept up to 1 sample of mislabeling at split transitions with
        sub-sample silence gaps (e.g. coarse-TR Fmri); otherwise raise.

    Invariants
    ----------
    - Every emitted chunk has ``duration >= min_duration``.
    - Every emitted chunk is label-homogeneous when ``event_type_to_split_by`` is set.
    - ``"equal"`` is lossless: concatenation reconstructs the input event.
    - ``"max"`` may silently drop sections/trailing pieces shorter than ``min_duration``.

    Raises
    ------
    ValueError
        - ``tiling="equal"`` and a same-``split`` run shorter than ``min_duration``
          (cannot tile without losing labeled data — switch to ``tiling="max"``
          to drop short pieces instead).
        - Two consecutive differently-labeled runs are less than one
          sample apart (cannot separate without label leakage).

    Examples
    --------
    Simple chunking (each ``x`` = one sample; sound sampled at 1 Hz)::

        input:
            max_duration: 4
            events:
                sound:   [x x x x x x x x x x x x x]     # 13 s
        out (tiling="max"):   # tile with max duration + trail
            events:
                sound1:  [x x x x]
                sound2:          [x x x x]
                sound3:                  [x x x x]
                sound4:                          [x]     # short trailing chunk
        out (tiling="equal"):  # tile with ~ same length
            events:
                sound1:  [x x x]
                sound2:        [x x x]
                sound3:              [x x x x]           # 3.25 s ideal, rounded to whole sample
                sound4:                      [x x x]

    With train/test split labels::

        input:
            max_duration: 4
            event_type_to_split_by: Word
            events:
                sound:   [x x x x x x x x x x x x x]     # 13 s
                word:     1 1 1 - - 2 2 2 2 2 2 2 2      # 1=test, 2=train, -=silence
        out (tiling="equal"):                            # split-aligned, then sub-tiled
            events:
                sound1:  [x x x x]                       # test run
                sound2:          [x x x]                 # train run, 3 equal chunks
                sound3:                [x x x]
                sound4:                      [x x x]
    """

    event_type_to_chunk: str
    event_type_to_split_by: str | None = None
    min_duration: float = 0.0
    max_duration: float = np.inf
    tiling: Tiling = "max"
    allow_sample_leakage: bool = False

    # Slack for sample-aligned float compares; absorbs float noise without
    # masking real mismatches so long as ``event.frequency << 1 / tol``
    # (i.e. tol << one sample period). Enforced per-event in ``_chunk_timeline``.
    _SAMPLE_GRID_TOL: tp.ClassVar[float] = 1e-6

    def model_post_init(self, log__: object) -> None:
        super().model_post_init(log__)
        cls = ev.Event._CLASSES.get(self.event_type_to_chunk)
        if cls is None or not issubclass(cls, ev.BaseSplittableEvent):
            splittable = [
                n
                for n, c in ev.Event._CLASSES.items()
                if issubclass(c, ev.BaseSplittableEvent)
            ]
            raise ValueError(
                f"{self.event_type_to_chunk!r} is not a splittable event type. "
                f"Use one of {splittable}"
            )
        if (
            self.tiling == "equal"
            and 2 * self.min_duration > self.max_duration + self._SAMPLE_GRID_TOL
        ):
            raise ValueError(
                f"min_duration={self.min_duration} must be <= "
                f"max_duration/2={self.max_duration / 2} for tiling='equal' "
                f"so chunks always land in [min_duration, max_duration]"
            )

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.event_type_to_split_by is not None and "split" not in events.columns:
            raise RuntimeError("Events must have a split column")
        chunked = [self._chunk_timeline(df) for _, df in events.groupby("timeline")]
        return pd.concat(chunked).reset_index(drop=True)

    def _chunk_timeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one timeline's events with target rows replaced by their chunks."""
        to_chunk = df.type == self.event_type_to_chunk
        if not any(to_chunk):
            return df
        use_rows = (
            df.loc[df.type == self.event_type_to_split_by]
            if self.event_type_to_split_by is not None
            else None
        )
        rows = df.loc[to_chunk]
        added: list[dict] = []
        for row in rows.itertuples(index=False):
            event = ev.BaseSplittableEvent.from_dict(row)
            if event.frequency * self._SAMPLE_GRID_TOL >= 1:
                raise ValueError(
                    f"event frequency {event.frequency} Hz is too high for the "
                    f"sample-grid tolerance {self._SAMPLE_GRID_TOL}s "
                    f"(>= 1 sample period); boundary compares would mask real "
                    f"1-sample mismatches."
                )
            sections = _build_sections(event, use_rows, self.allow_sample_leakage)
            tiled = [
                sub
                for s in sections
                for sub in s.tile(self.min_duration, self.max_duration, self.tiling)
            ]
            rel_tps = [t.start - event.start for t in tiled[1:]]
            pieces = event._split(rel_tps)
            added.extend(
                p.to_dict()
                for p in pieces
                if p.duration >= self.min_duration - self._SAMPLE_GRID_TOL
            )
        return pd.concat([df.drop(rows.index), pd.DataFrame(added)])
