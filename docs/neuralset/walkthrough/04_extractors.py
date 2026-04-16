"""
.. _tuto_extractors:

Extractors
==========

An :term:`extractor <Extractor>` turns events into
tensors for a given time window.  Where a :term:`Study` produces an
events DataFrame and :term:`transforms <EventsTransform>` refine it,
extractors are the step that produces the numerical data your model
consumes.

.. note::

   This tutorial uses spacy embeddings, included in
   ``pip install 'neuralset[tutorials]'`` (see :ref:`tuto_studies`).
"""

# %%
# Calling an Extractor
# --------------------
#
# An extractor is called with ``(events, start, duration)`` and returns
# a :class:`torch.Tensor`.  Let's load some events and try the
# ``MegExtractor``, which reads MEG channels at a given frequency:

import neuralset as ns

study = ns.Study(name="Fake2025Meg", path=ns.CACHE_FOLDER, query="timeline_index < 1")
events = study.run()

# %%

meg = ns.extractors.MegExtractor(frequency=100.0)

sample = meg(events, start=0.0, duration=1.0)
print(f"MEG shape: {sample.shape}  (channels x time)")

# %%
# Aggregation
# -----------
#
# An extractor is called with a time window (``start``, ``duration``).
# If that window contains **several matching events** — e.g. three
# words in a 2-second window — the extractor needs a rule for combining
# them.  That rule is set by ``aggregation``:
#
# - ``"single"`` (default) — expects exactly one event; raises otherwise.
#   The most common mode for MEG, fMRI, etc. when each segment targets
#   one stimulus.
# - ``"first"`` / ``"last"`` — picks one event by position.
# - ``"mean"`` / ``"sum"`` — element-wise average or sum across events.
#   Useful when multiple words fall within the same fMRI TR, or for
#   averaging aligned neural recordings across subjects.
# - ``"trigger"`` — extract only the event that *anchors* the segment
#   (its trigger), ignoring other events in the window.  Useful for
#   static extractors (``frequency=0``) on Word or Image events: you
#   want the feature of the trigger word, not of every word in the
#   window.  Not meaningful for ``frequency > 0`` extractors.
#   The :doc:`segmenter tutorial <05_segmenter>` explains how triggers
#   define segments.

import pandas as pd

two_events = ns.events.standardize_events(
    pd.DataFrame(
        [
            dict(type="Word", start=10.0, duration=0.3, text="hello", timeline="run-01"),
            dict(type="Word", start=11.0, duration=0.4, text="world", timeline="run-01"),
        ]
    )
)

pulse_mean = ns.extractors.Pulse(aggregation="mean")
print(f"Mean of 2 pulses: {pulse_mean(two_events, start=9.5, duration=2.0)}")

pulse_first = ns.extractors.Pulse(aggregation="first")
print(f"First pulse only: {pulse_first(two_events, start=9.5, duration=2.0)}")

# %%
# Computation and caching
# -----------------------
#
# Many extractors involve expensive operations (loading a model, computing
# embeddings for every event).  The
# :meth:`~neuralset.extractors.BaseExtractor.prepare` method
# precomputes these results for all events up front so that subsequent
# calls are fast.  In practice you rarely call it yourself — the
# :class:`~neuralset.dataloader.Segmenter` calls it automatically.
#
# **Disk caching** — without ``infra``, results live in memory only and
# are lost between runs.  Pass an ``infra`` with a cache folder to
# persist them to disk:
#
# .. code-block:: python
#
#    # Requires: `pip install transformers` (or `pip install "neuralset[all]"`)
#    extractor = HuggingFaceImage(
#        model_name="facebook/dinov2-small",
#        infra={"folder": "/path/to/cache"},
#    )
#    extractor.prepare(events)   # computes & saves to /path/to/cache
#    vec = extractor(events, start=0, duration=1.0)  # reads from cache
#
# **Cluster execution** — heavy extractors (large language models, image
# encoders) can be dispatched to a Slurm cluster so each event is
# processed on a dedicated GPU worker:
#
# .. code-block:: python
#
#    # Requires: `pip install transformers` (or `pip install "neuralset[all]"`)
#    extractor = HuggingFaceText(
#        model_name="gpt2",
#        infra={"backend": "Slurm", "folder": "/cache",
#               "gpus_per_node": 1, "partition": "gpu"},
#    )
#    extractor.prepare(events)  # submits jobs, waits for results
#
# ``"SubmititDebug"`` runs the same code path inline, which is useful
# for local debugging.
#
# See :doc:`/neuralset/caching_and_cluster` for the full guide on
# backends, cache modes, and cluster configuration.

# %%
# TimedArray
# ----------
#
# Internally, each extractor converts an event into a
# :class:`~neuralset.base.TimedArray` — a container holding a numpy
# array alongside its ``start``, ``duration``, and ``frequency``. The
# extractor's aggregation logic then combines multiple TimedArrays into
# the final output tensor.
#
# You rarely interact with TimedArray directly, but it is useful when
# writing custom extractors (see `Create Your Own`_ below).

# %%
# Output Shapes
# -------------
#
# **No time dimension** — when ``frequency=0`` the extractor produces a
# tensor whose shape doesn't depend on segment duration (most commonly
# a 1-d vector, but the shape is arbitrary):

# Requires: `pip install spacy` (or `pip install "neuralset[all]"`) and
# `python -m spacy download en_core_web_lg`  — ~400 MB, auto-downloaded on first run.
emb = ns.extractors.SpacyEmbedding(language="english", aggregation="mean")
emb.prepare(two_events)
static_result = emb(two_events, start=9.5, duration=2.0)
print(f"SpacyEmbedding(frequency=0): shape {static_result.shape}")

# %%
# **With a time dimension** — the same extractor with ``frequency > 0``
# broadcasts its value across the time axis.  This is especially useful
# for word features — it aligns them on the same time grid as neural
# data so they can be combined directly:

emb_timed = ns.extractors.SpacyEmbedding(
    language="english", frequency=100.0, aggregation="mean"
)
emb_timed.prepare(two_events)
timed_result = emb_timed(two_events, start=9.5, duration=2.0)
print(f"SpacyEmbedding(frequency=100): shape {timed_result.shape}  (features x time)")

# %%
# Neural extractors like ``MegExtractor`` are inherently time-varying —
# they always require ``frequency > 0``:

print(f"MegExtractor(frequency=100): shape {sample.shape}  (channels x time)")

# %%
# EEG / MEG / EMG
# ~~~~~~~~~~~~~~~~
#
# Neural recordings are handled by extractors inheriting from
# :class:`~neuralset.extractors.neuro.MneRaw`. The processing pipeline
# includes channel picks, filtering, resampling, scaling, and baseline
# correction — all configured through extractor parameters.
#
# .. code-block:: python
#
#    from neuralset.extractors import EegExtractor
#
#    eeg = EegExtractor(
#        frequency=100,             # resample to 100 Hz
#        filter=(0.5, 30),          # bandpass filter
#        scaler="robust",           # per-channel scaling
#        baseline=0.3,              # 300 ms baseline correction
#    )
#    # For a 2-second segment with 64 EEG channels:
#    # result.shape -> (64, 200)
#
# .. warning::
#
#    Baseline correction is relative to the **segment start**,
#    not the epoch onset. See :class:`~neuralset.extractors.neuro.MneRaw`
#    for details.

# %%
# iEEG
# ~~~~~
#
# :class:`~neuralset.extractors.neuro.IeegExtractor` handles intracranial
# EEG (sEEG, ECoG). It supports bipolar referencing, where each channel
# is referenced to the next electrode on the same shank. If a reference
# electrode is missing, it silently falls back to the next available one.
# See :class:`~neuralset.extractors.neuro.IeegExtractor` for channel
# naming conventions.

# %%
# fNIRS
# ~~~~~~
#
# :class:`~neuralset.extractors.neuro.FnirsExtractor` implements a
# multi-step preprocessing pipeline controlled by boolean flags:
#
# 1. Convert to optical density
# 2. Apply Beer-Lambert law (requires step 1)
# 3. Apply TDDR artifact correction (requires step 2)
#
# These flags form a dependency chain — enabling step 3 without step 1
# will raise an error. Requires ``mne-nirs``.

# %%
# fMRI
# ~~~~~
#
# :class:`~neuralset.extractors.neuro.Fmri` supports two mutually
# exclusive modes:
#
# - **mesh** — surface-based extraction
# - **atlas** — parcellation-based extraction
#
# Additional parameters: ``space`` (e.g. ``"MNI152NLin2009cAsym"``),
# ``confounds_strategy`` for nuisance regression, and ``offset`` to
# account for hemodynamic delay.

# %%
# Text
# ~~~~~
#
# :class:`~neuralset.extractors.text.HuggingFaceText` extracts
# contextualized embeddings from transformer models. It uses the
# ``context`` field on Word events (populated by text transforms)
# to provide surrounding context for each word.
#
# Layer and token aggregation are configurable via
# :class:`~neuralset.extractors.base.HuggingFaceMixin`:
#
# .. code-block:: python
#
#    from neuralset.extractors import HuggingFaceText
#
#    text_ext = HuggingFaceText(
#        model_name="gpt2",
#        layers=(8, 9, 10, 11),     # which transformer layers to use
#        token_aggregation="mean",  # mean over subword tokens
#    )
#
# :class:`~neuralset.extractors.text.SpacyEmbedding` provides lighter
# static embeddings that don't require a GPU.

# %%
# Image
# ~~~~~~
#
# :class:`~neuralset.extractors.image.HuggingFaceImage` extracts image
# embeddings using HuggingFace vision models (DINOv2, CLIP, etc.).
# Classic feature extractors (:class:`~neuralset.extractors.image.RFFT2D`,
# :class:`~neuralset.extractors.image.HOG`,
# :class:`~neuralset.extractors.image.LBP`) are also available.
#
# .. note::
#
#    The ``imsize`` parameter may be overridden by the HuggingFace
#    processor's expected input size.

# %%
# Audio
# ~~~~~~
#
# For audio, **dedicated subclasses must be used** for specific models:
#
# - :class:`~neuralset.extractors.audio.Whisper` for OpenAI Whisper
# - :class:`~neuralset.extractors.audio.SeamlessM4T` for Meta SeamlessM4T
# - :class:`~neuralset.extractors.audio.Wav2VecBert` for Wav2Vec-BERT
#
# Do **not** use the base ``HuggingFaceAudio`` class for these models —
# they require model-specific preprocessing.
#
# ``frequency="native"`` uses the audio file's native sample rate, which
# is an approximation — set an explicit frequency when possible.

# %%
# Create Your Own
# ---------------
#
# Subclass :class:`~neuralset.extractors.BaseStatic` and implement
# ``get_static(event) -> Tensor``.  Set ``event_types`` to control
# which events the extractor responds to.
#
# Here we compute the character count of each Word event:

import typing as tp

import torch

from neuralset import extractors as ext_mod


class CharCount(ext_mod.BaseStatic):
    event_types: tp.Literal["Word"] = "Word"

    def get_static(self, event: "ns.events.etypes.Event") -> torch.Tensor:
        return torch.tensor([len(event.text)], dtype=torch.float32)


custom_events = ns.events.standardize_events(
    pd.DataFrame(
        [
            dict(type="Word", start=1.0, duration=0.3, text="hello", timeline="run-01"),
            dict(
                type="Image",
                start=2.0,
                duration=1.0,
                filepath="stim_*.png",  # glob pattern — file need not exist
                timeline="run-01",
            ),
        ]
    )
)

extractor = CharCount()
custom_result = extractor(custom_events, start=0.5, duration=2.0)
print(f"CharCount result: {custom_result}  shape: {custom_result.shape}")

# %%
# The ``Image`` event is ignored because ``event_types`` is ``"Word"``.
# Only matching events are processed by the extractor.

# %%
# With ``frequency > 0`` the result becomes a matrix (time x features):
#
# .. code-block:: python
#
#    extractor_dyn = CharCount(frequency=100, aggregation="mean")
#    mat = extractor_dyn(custom_events, start=0.5, duration=2.0)
#    print(mat.shape)  # (1, 200)

# %%
# Next Steps
# ----------
#
# - Segment events into a dataset: :doc:`05_segmenter`
# - Compose full pipelines: :doc:`06_chains`
#
# .. seealso::
#
#    - :doc:`/neuralset/reference/reference` -- full API for all extractors
