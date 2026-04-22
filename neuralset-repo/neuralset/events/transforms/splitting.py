# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import typing as tp
from collections import Counter

import numpy as np
import pandas as pd
import pydantic
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from neuralset import base, extractors

from .. import etypes as ev
from ..study import EventsTransform
from . import text as _text
from . import utils as _utils

logger = logging.getLogger(__name__)


class AssignSentenceSplit(EventsTransform):
    """Assign a train/val/test split to each sentence in a deterministic fashion."""

    min_duration: float | None = None
    min_words: int | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    max_unmatched_ratio: float = 0.0

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["max_unmatched_ratio"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not sum(self.ratios) == 1:
            raise ValueError(f"Split ratios must sum to 1, got {sum(self.ratios)}")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "timeline" not in events.columns:
            events["timeline"] = "#foo#"
        wtypes = ev.EventTypesHelper("Word")
        words_df = events.loc[events.type.isin(wtypes.names), :]
        n_words = len(words_df)
        ratio = sum(not s or not isinstance(s, str) for s in words_df.sentence) / n_words
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            msg = f"Ratio of words with no sentence match is {ratio:.2f} while {cls}.{max_unmatched_ratio=}"
            raise RuntimeError(msg)
        sentences = _utils._extract_sentences(events)
        merged = _utils._merge_sentences(
            sentences, min_duration=self.min_duration, min_words=self.min_words
        )
        ratios = dict(train=self.ratios[0], val=self.ratios[1], test=self.ratios[2])
        ratios = {x: y for x, y in ratios.items() if y > 0}
        if len(ratios) == 1:
            events.loc[events.type.isin(wtypes.names), "split"] = list(ratios)[0]
            if tuple(events.timeline.unique()) == ("#foo#",):
                events = events.drop("timeline", axis=1)
            return events
        splitter = _utils.DeterministicSplitter(ratios, seed=self.seed)
        undef = "undefined"
        affectations: dict[str | float, tuple[str, ...] | str] = {
            _utils.MISSING_SENTENCE: undef
        }
        groups: dict[str, set[str]] = {}
        for part in merged:
            string = "".join(s.text for s in part)
            if string not in affectations:
                affectations[string] = splitter(string)
            split = affectations[string]
            for seq in part:
                groups.setdefault(seq.text, set()).add(string)
                if affectations.setdefault(seq.text, split) != split:
                    affectations[seq.text] = undef
                    conflicts = groups[seq.text]
                    logger.warning(
                        'Sequence split "%s" set to undefined because it belongs to conflicting groups: %s',
                        seq.text,
                        conflicts,
                    )
        valid = ~(np.logical_or(events.sentence.isnull(), events.sentence == ""))
        splits = events.loc[valid].sentence.apply(str).apply(lambda x: affectations[x])
        events.loc[valid, "split"] = splits
        is_word = events.type.isin(wtypes.names)
        events.loc[np.logical_and(~valid, is_word), "split"] = undef
        if tuple(events.timeline.unique()) == ("#foo#",):
            events = events.drop("timeline", axis=1)
        return events


class AssignKSplits(EventsTransform):
    """Assign k splits to events."""

    k: int
    groupby: str | None = None

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.groupby is not None:
            groups = []
            sub = self.model_validate({"groupby": None, "k": self.k})
            for _, g in events.groupby(self.groupby, sort=False):
                groups.append(sub._run(g))
            return pd.concat(groups, ignore_index=True)
        if "timeline" in events.columns and len(events.timeline.unique()) > 1:
            timelines = []
            groups = [g for _, g in events.groupby("timeline", sort=False)]
            sub_ks = np.ones(len(groups))
            if self.k > len(groups):
                to_add = self.k - len(groups)
                sub_ks += np.diff(
                    [np.round(to_add * g / len(groups)) for g in range(len(groups) + 1)]
                )
            for sub_k, subevents in zip(sub_ks, groups):
                sub = self.model_validate({"k": int(sub_k)})
                timelines.append(sub._run(subevents))
            return pd.concat(timelines, ignore_index=True)
        events = events.copy(deep=True)
        events.loc[:, "split"] = ""
        sent_df = events.loc[events.type == "Sentence", :]
        hashed = hashlib.sha256(" ".join(sent_df.text).encode()).hexdigest()[:8]
        if len(sent_df) < self.k:
            msg = f"Not enough sentences for {self.k} splits: {sent_df.text}"
            raise RuntimeError(msg)
        start = min(events.start) - 1
        wtypes = ev.EventTypesHelper("Word")
        for n in range(self.k):
            ind = int(np.round((len(sent_df) - 1) * (n + 1) / self.k))
            last = sent_df.iloc[ind]
            stop = last.duration + last.start
            select: tp.Any = events.type.isin(wtypes.names)
            select = np.logical_and(select, events.start >= start)
            select = np.logical_and(select, events.start < stop)
            events.loc[select, "split"] = f"{hashed}-ksplit{n + 1}/{self.k}"
            start = stop
        return events


class AssignWordSplitAndContext(EventsTransform):
    """Sequence of 3 standard operations: add sentences, assign split, add context.

    .. warning::
       **Unstable API** — the context step will change in a future release.
    """

    min_duration: float | None = None
    min_words: int | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    max_unmatched_ratio: float = 0.0
    sentence_only: bool = True
    max_context_len: int | None = None
    override_sentences: bool = False
    split_field: str = "split"

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["max_unmatched_ratio"]

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        params = [(x, getattr(self, x)) for x in type(self).model_fields]
        names = ["sentence", "split", "context"]
        transforms = [self.transform(name) for name in names]  # type: ignore
        add_sentence, assign_split, add_context = transforms
        expected = {name for e in transforms for name in type(e).model_fields}
        diff = expected.symmetric_difference({p[0] for p in params})
        if diff:
            raise RuntimeError(
                f"Mismatch between proposed parameters and transforms parameters: {diff}"
            )
        df = add_sentence._run(events)
        df = assign_split._run(df)
        df = add_context._run(df)
        return df

    def transform(
        self, name: tp.Literal["sentence", "split", "context"]
    ) -> EventsTransform:
        params = [(x, getattr(self, x)) for x in type(self).model_fields if x != "type"]
        transforms: dict[str, tp.Type[EventsTransform]] = {
            "sentence": _text.AddSentenceToWords,
            "split": AssignSentenceSplit,
            "context": _text.AddContextToWords,
        }
        cls = transforms[name]
        return cls(**{x: y for x, y in params if x in cls.model_fields})  # type: ignore


class SklearnSplit(EventsTransform):
    """Perform train/val/test split using sklearn's ``train_test_split``.

    Parameters
    ----------
    split_by : str
        Column name to use for splitting by (e.g., 'timeline' or 'subject'). If set to "_index" and
        "_index" is not in the events dataframe, the events dataframe will be reset to have a new
        column with row indices, named "_index".
    valid_split_ratio : float
        Ratio of the full dataset to use for validation.
    test_split_ratio : float
        Ratio of the full dataset to use for testing.
    valid_random_state : int
        Random state for validation split.
    test_random_state : int
        Random state for test split.
    stratify_by : str | None
        Column name to use for stratified splitting. If None, no stratification is applied.
    """

    split_by: str = "timeline"
    valid_split_ratio: float = pydantic.Field(
        0.2, strict=True, ge=0.0, le=1.0, allow_inf_nan=False
    )
    test_split_ratio: float = pydantic.Field(
        0.2, strict=True, ge=0.0, le=1.0, allow_inf_nan=False
    )
    valid_random_state: int = 33
    test_random_state: int = 33
    stratify_by: str | None = None

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        total = self.valid_split_ratio + self.test_split_ratio
        if total > 1.0:
            raise ValueError(
                f"valid_split_ratio + test_split_ratio must be <= 1.0, got {total}"
            )

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.split_by == "_index" and "_index" not in events.columns:
            events = (
                events.copy().reset_index(drop=False).rename(columns={"index": "_index"})
            )
        cols = [self.split_by]
        if self.stratify_by is not None:
            cols.append(self.stratify_by)
        groups = events[cols].dropna().drop_duplicates()
        if self.stratify_by is not None and groups[self.split_by].duplicated().sum() != 0:
            raise ValueError("Stratified splitting requires unique values in `split_by`.")

        inds_train_valid, inds_test = train_test_split(
            groups.index,
            test_size=self.test_split_ratio,
            random_state=self.test_random_state,
            shuffle=True,
            stratify=groups[self.stratify_by] if self.stratify_by is not None else None,
        )
        groups_train_valid = groups.loc[inds_train_valid]

        adjusted_valid_split_ratio = self.valid_split_ratio / (
            1.0 - self.test_split_ratio
        )
        inds_train, inds_valid = train_test_split(
            groups_train_valid.index,
            test_size=adjusted_valid_split_ratio,
            random_state=self.valid_random_state,
            shuffle=True,
            stratify=(
                groups_train_valid[self.stratify_by]
                if self.stratify_by is not None
                else None
            ),
        )

        split_mapping = {
            **{k: "train" for k in events.loc[inds_train, self.split_by].unique()},
            **{k: "val" for k in events.loc[inds_valid, self.split_by].unique()},
            **{k: "test" for k in events.loc[inds_test, self.split_by].unique()},
        }
        events["split"] = events[self.split_by].map(split_mapping)
        return events


class SimilaritySplitter(base.BaseModel):
    """A class used to split events based on similarity clustering of static extractors.
    The class uses agglomerative clustering on precomputed embeddings of the events
    to ensure that same and similar events remain in the same split to avoid data leaking.

    Parameters
    ----------
    extractor : BaseStatic
        A static feature extraction model that defines the type of event
        and provides methods to extract embeddings from events.
    ratios : Dict[str, float]
        A dictionary defining the proportion of events for each split.
        The sum of all ratios must equal 1.
    threshold : float
        The threshold for the distance used in the agglomerative clustering.
        Events with a distance below this threshold are grouped into clusters.
    norm_feats : bool
        If True, the extractor embeddings are normalized before computing
        the cosine similarity matrix.

    """

    extractor: extractors.BaseStatic
    ratios: dict[str, float] = {"train": 0.5, "val": 0.25, "test": 0.25}
    threshold: float = 0.2
    norm_feats: bool = True

    def model_post_init(self, log__) -> None:
        super().model_post_init(log__)
        if any(ratio <= 0 for ratio in self.ratios.values()):
            msg = f"All ratios must be greater than 0. Got: {self.ratios}"
            raise ValueError(msg)

        total_ratio = sum(self.ratios.values())
        if not np.isclose(total_ratio, 1.0, atol=1e-8):
            msg = f"The sum of ratios must be equal to 1.0. Got: {total_ratio}"
            raise ValueError(msg)

    def _similarity_clustering(self, embeddings) -> list[int]:
        """Perform similarity-based clustering on the similarity matrix.

        Parameters
        ----------
        similarity_matrix: np.ndarray
            Precomputed cosine similarity matrix of dimension (number of events, number of events).


        Returns
        -------
        List[int]
            Clusters of indices.
        """

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        if self.norm_feats:
            embeddings = torch.from_numpy(embeddings)
            embeddings = F.layer_norm(embeddings, (embeddings.size(-1),)).numpy()

        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            distance_threshold=self.threshold,
            linkage="complete",
        )

        return [label.item() for label in clustering.fit_predict(distance_matrix)]

    def _cluster_assignment(self, clusters: list[int]) -> list[str]:
        """Assigns clusters to predefined splits (e.g., 'train', 'val', 'test') based on the
        ratios specified in `self.ratios`. Each cluster is assigned to a split such that
        the number of clusters in each split respects the specified ratio, and clusters are
        entirely allocated to a single split (no partial assignments).

        Parameters
        ----------
        clusters: list[int]
            A list of cluster IDs, where each cluster is represented by an integer,
            and the list contains the clusters to be assigned to splits.

        Returns
        -------
        list[str]
            A list of split labels ('train', 'val', 'test', etc.), corresponding
            to the same length as `clusters`, where each element indicates the split
            assignment for the corresponding cluster in the `clusters` input.
        """
        cluster_count = Counter(clusters)
        total_count = len(clusters)
        sorted_splits = sorted(self.ratios.items(), key=lambda x: x[1])
        split_sizes = {}
        cluster_split = {}

        logger.info(f"{len(cluster_count)} clusters found for {total_count} events.")

        remaining_count = total_count
        for split, ratio in sorted_splits[:-1]:
            split_sizes[split] = int(np.ceil(ratio * total_count))
            remaining_count -= int(np.ceil(ratio * total_count))
        largest_split = sorted_splits[-1][0]
        split_sizes[largest_split] = remaining_count

        for key, count in cluster_count.items():
            for split, size in split_sizes.items():
                if size >= count:
                    cluster_split[key] = split
                    split_sizes[split] -= count
                    break
            if key not in cluster_split:
                msg = f"Not enough space in splits to assign cluster {key} with {count} elements to one split. "
                msg += f"Assigning it to largest split which is {largest_split}."
                logger.warning(msg)
                cluster_split[key] = largest_split

        if len(cluster_split) == 0:
            msg = "All examples are assigned to the same cluster. "
            msg += f"Try modifying the threshold value (currently {self.threshold}) to get more clusters."
            raise ValueError(msg)

        if not (set(cluster_split.values()) == set(self.ratios.keys())):
            msg = "Some splits have no clusters. Change the ratios or increase the number of examples."
            raise ValueError(msg)

        assigned_splits = [cluster_split[c] for c in clusters]
        counts = Counter(assigned_splits)
        achieved = {k: counts[k] / total_count for k in self.ratios}
        max_dev = max(abs(achieved[k] - self.ratios[k]) for k in self.ratios)

        if max_dev > 0.1:
            msg = f"Requested split ratios={ {k: f'{v:.2f}' for k, v in self.ratios.items()} }, "
            msg += (
                f"but achieved ratios={ {k: f'{achieved[k]:.2f}' for k in self.ratios} }."
            )
            msg += (
                "Consider lowering the clustering threshold or rebalancing the dataset."
            )
            logger.warning(msg)

        return assigned_splits

    def compute_clusters(self, events: pd.DataFrame) -> pd.DataFrame:
        """Filter events by the event types defined in the extractor and compute cluster assignments."""
        self.extractor.prepare(events)
        splitted_events = events[
            events["type"].isin(ev.EventTypesHelper(self.extractor.event_types).names)
        ].copy()

        embedding_list = []
        for _, row in splitted_events.iterrows():
            e = ev.Event.from_dict(row.to_dict())
            embedding_list.append(self.extractor.get_static(e).flatten())

        embeddings = torch.stack(embedding_list).numpy()
        clusters = self._similarity_clustering(embeddings)

        counts = pd.Series(clusters).value_counts().values
        if max(counts) / min(counts) >= 10.0:
            msg = "The clusters are highly imbalanced. Try lowering the threshold value "
            msg += f"(currently {self.threshold}) to get more clusters."
            logger.warning(msg)

        splitted_events["cluster_id"] = clusters
        return splitted_events

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        """Splits a given DataFrame based on similarity clustering.

        Parameters
        ----------
        events: pd.DataFrame
            A DataFrame containing event data, with each row representing
            a single event and columns representing the event's attributes.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with an additional 'split' column, which
            indicates the assigned split for each event.

        """
        splitted_events = self.compute_clusters(events)
        cluster_assignment = self._cluster_assignment(
            splitted_events["cluster_id"].tolist()
        )

        out_events = events.copy()
        out_events["split"] = ""
        out_events.loc[splitted_events.index, "split"] = cluster_assignment

        return out_events
