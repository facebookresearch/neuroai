# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified model registry for plotting, table generation, and parameter counting.

This module is the single source of truth for per-model metadata used by:

* :mod:`neuralbench.plots._constants`, which derives ``MODEL_PARAMS``,
  ``MODEL_YEAR``, ``PRETRAINING_OVERLAP``, ``MEEG_CLASSIC_DISPLAY``,
  ``FMRI_CLASSIC_DISPLAY``, ``MEEG_FM_DISPLAY``, ``FMRI_FM_DISPLAY``,
  ``CLASSIC_DISPLAY``, ``FM_DISPLAY`` and the registry portion of
  ``MODEL_DISPLAY_NAMES`` from :data:`MODELS`.
* ``brainai-repo/brainai/bench/scripts/count_model_params.py``, which
  iterates :data:`MODELS` to instantiate each model via its
  :attr:`ModelEntry.builder`, applies :attr:`ModelEntry.backbone_subtract`,
  and writes the resulting integer counts to
  ``neuralbench/plots/model_params.json``.
* ``brainai-repo/brainai/bench/scripts/generate_model_table.py``, which
  iterates :data:`MODELS` filtered by ``family`` / ``device`` to render the
  paper LaTeX tables.

Cached parameter counts live in :data:`PARAM_COUNT_CACHE_PATH`
(``model_params.json`` next to this file) and are loaded lazily by
:func:`load_param_counts`.  Re-run the count script to refresh the cache.

``builder`` callables are imported lazily inside each builder so that simply
importing this module (e.g. from plotting code) does not pull in ``torch``
and ``braindecode``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Representative shape used for "backbone" parameter counting of task-specific
# braindecode models.  See count_model_params.py for the rationale.
# ---------------------------------------------------------------------------

_REPR_N_CHANS = 22
_REPR_N_TIMES = 256
_REPR_N_OUTPUTS = 4


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """Metadata for a single model in the NeuralBench plotting / paper stack.

    Parameters
    ----------
    name
        Display name (e.g. ``"BENDR"``).  This is the key used by every
        downstream constant (``MODEL_PARAMS``, ``MODEL_YEAR``,
        ``MEEG_CLASSIC_DISPLAY``, ...).
    family
        ``"classic"`` for end-to-end task models, ``"foundation"`` for
        pretrained EEG foundation models.
    device
        Recording modality the model targets.
    year
        Publication year.  ``None`` for entries (e.g. baselines) that do not
        appear in :data:`neuralbench.plots._constants.MODEL_YEAR`.
    bibtex
        BibTeX citation key (used by the LaTeX tables).
    config_name
        ``brain_model_config.name`` as written in the neuralbench YAML
        configs (e.g. ``"NtBendr"``).  Used to derive
        :data:`neuralbench.plots._constants.MODEL_DISPLAY_NAMES`.
    cli_name
        Slug-style identifier used as a CLI argument and as the YAML
        filename stem in ``neuralbench/models/`` (e.g. ``"bendr"``,
        ``"shallow_fbcsp_net"``).  Used to derive
        :data:`neuralbench.registry.CLASSIC_MODELS` /
        :data:`neuralbench.registry.FM_MODELS` and the model dicts in
        ``brainai-repo/brainai/bench/scripts/generate_results_json.py``.
    variant
        Foundation-model variant label for the LaTeX table (e.g. ``"Base"``,
        ``"Large"``, ``"---"``).
    pretrain_strategy
        Short description of the pretraining objective (foundation only).
    pretrain_data
        Short description of the pretraining corpus (foundation only).
    pretraining_overlap
        Set of ``neuralhub`` study class names the model was pretrained on.
        Drives downstream-bar hatching via
        :data:`neuralbench.plots._constants.PRETRAINING_OVERLAP`.
    builder
        Zero-argument callable returning an instantiated ``nn.Module`` for
        parameter counting.  Imports are done lazily inside the callable.
    backbone_subtract
        Tuple of dotted attribute paths whose parameters are subtracted from
        the total to obtain the "backbone" count.  Empty tuple = count the
        whole model.
    """

    name: str
    family: Literal["classic", "foundation"]
    device: Literal["eeg", "fmri"]
    bibtex: str | None = None
    year: int | None = None
    config_name: str | None = None
    cli_name: str | None = None
    variant: str | None = None
    pretrain_strategy: str | None = None
    pretrain_data: str | None = None
    pretraining_overlap: frozenset[str] | None = None
    builder: Callable[[], "torch.nn.Module"] | None = None
    backbone_subtract: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Builder helpers (lazy imports)
# ---------------------------------------------------------------------------


def _make_classic_braindecode_builder(cls_name: str) -> Callable[[], "torch.nn.Module"]:
    def _build() -> "torch.nn.Module":
        import braindecode.models as bdm

        cls = getattr(bdm, cls_name)
        return cls(
            n_chans=_REPR_N_CHANS,
            n_times=_REPR_N_TIMES,
            n_outputs=_REPR_N_OUTPUTS,
        )

    return _build


def _build_simpleconv_time_agg() -> "torch.nn.Module":
    import torch

    from neuraltrain.models.common import ChannelMerger, FourierEmb, Mlp
    from neuraltrain.models.simpleconv import SimpleConvTimeAgg

    config = SimpleConvTimeAgg(
        hidden=320,
        depth=4,
        linear_out=False,
        kernel_size=3,
        dilation_period=5,
        skip=True,
        glu=2,
        glu_context=1,
        gelu=True,
        batch_norm=True,
        merger_config=ChannelMerger(
            n_virtual_channels=270,
            fourier_emb_config=FourierEmb(
                n_freqs=None,
                total_dim=2000,
                n_dims=3,
            ),
            dropout=0.2,
            per_subject=False,
            embed_ref=False,
        ),
        initial_linear=270,
        backbone_out_channels=512,
        time_agg_out="linear",
        output_head_config=Mlp(hidden_sizes=[-1]),
    )
    model = config.build(n_in_channels=_REPR_N_CHANS, n_outputs=_REPR_N_OUTPUTS)
    # Materialise lazy layers via a forward pass.
    with torch.no_grad():
        model.eval()
        x = torch.randn(1, _REPR_N_CHANS, _REPR_N_TIMES)
        pos = torch.randn(1, _REPR_N_CHANS, 3)
        model(x, channel_positions=pos)
    return model


def _build_bendr() -> "torch.nn.Module":
    import braindecode.models as bdm

    # Architectural instantiation only (no pretrained-weight loading): the
    # parameter count is independent of loaded weights, and the
    # "braindecode-bendr" checkpoint has 20-channel input weights that would
    # mismatch n_chans=19 under recent braindecode versions. The benchmark's
    # actual config uses the 19-channel input via the wrapper-level channel
    # adapter, which is what this counts (the encoder's 19-channel input
    # weights only differ from the 20-channel checkpoint by 1,536 params).
    return bdm.BENDR(
        n_chans=19,
        n_times=1536,
        n_outputs=1,
        final_layer=False,
    )


def _build_biot() -> "torch.nn.Module":
    import braindecode.models as bdm

    return bdm.BIOT.from_pretrained(
        "braindecode/biot-pretrained-six-datasets-18chs",
        n_times=1000,
        return_feature=True,
    )


def _build_labram() -> "torch.nn.Module":
    import braindecode.models as bdm

    return bdm.Labram.from_pretrained(
        "braindecode/labram-pretrained",
        n_times=3000,
        n_chans=128,
        n_outputs=0,
    )


def _build_cbramod() -> "torch.nn.Module":
    import braindecode.models as bdm

    return bdm.CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        return_encoder_output=True,
    )


def _build_reve() -> "torch.nn.Module":
    import braindecode.models as bdm

    return bdm.REVE.from_pretrained(
        "brain-bzh/reve-base",
        n_chans=_REPR_N_CHANS,
        n_times=_REPR_N_TIMES,
        n_outputs=2,
    )


def _build_luna() -> "torch.nn.Module":
    import braindecode.models as bdm

    return bdm.LUNA(
        n_chans=_REPR_N_CHANS,
        n_times=_REPR_N_TIMES,
        n_outputs=1,
        patch_size=40,
        num_queries=6,
        embed_dim=96,
        depth=10,
        num_heads=2,
        mlp_ratio=4.0,
        drop_path=0.1,
    )


# ---------------------------------------------------------------------------
# Pretraining-overlap sets
#
# These are duplicated here from neuralbench/refs to keep a single source of
# truth for foundation-model metadata.  See the original docstring in
# _constants.py for citations and rationale.
# ---------------------------------------------------------------------------

_PT_BENDR = frozenset(
    {
        "Lopez2017Tuab",
        "Hamid2020Tuar",
        "Harati2015Tuev",
        "Veloso2017Tuep",
        "Shah2018Tusz",
        "VonWeltin2017Tusl",
    }
)
_PT_CBRAMOD = frozenset(
    {
        "Lopez2017Tuab",
        "Hamid2020Tuar",
        "Harati2015Tuev",
        "Veloso2017Tuep",
        "Shah2018Tusz",
        "VonWeltin2017Tusl",
    }
)
_PT_LABRAM = frozenset(
    {
        "Schalk2004Bci",
        "Hamid2020Tuar",
        "Veloso2017Tuep",
        "Shah2018Tusz",
        "VonWeltin2017Tusl",
        "Detti2020Eeg",
    }
)
_PT_BIOT = frozenset(
    {
        "Dan2023Bids",
        "Lopez2017Tuab",
        "Harati2015Tuev",
    }
)
_PT_LUNA = frozenset(
    {
        "Harati2015Tuev",
        "Detti2020Eeg",
    }
)
_PT_REVE = frozenset(
    {
        "Hamid2020Tuar",
        "Veloso2017Tuep",
        "Shah2018Tusz",
        "VonWeltin2017Tusl",
        "Detti2020Eeg",
        "Shirazi2024Hbn",
        "Barachant2012Robust",
        "Cho2017Supporting",
        "Faller2012Autocalibration",
        "Kalunga2016Online",
        "Korczowski2019BrainBi2014A",
        "Korczowski2019BrainBi2014B",
        "Korczowski2019BrainBi2015A",
        "Korczowski2019BrainBi2015B",
        "Lee2019EegErp",
        "Lee2019EegMi",
        "Lee2019EegSsvep",
        "Liu2024Eeg",
        "Nakanishi2015Comparison",
        "Ofner2017Upper",
        "Schirrmeister2017Deep",
        "Shin2017OpenA",
        "Shin2017OpenB",
        "Sosulski2019Electroencephalogram",
        "Zhou2016Fully",
        "Hoffmann2008Efficient",
        "Arico2014Influence",
        "Khan2022Nmt",
        "VanDijk2022Two",
        "Gifford2022Large",
        "Accou2024Sparrkulee",
    }
)


# ---------------------------------------------------------------------------
# The registry
#
# Order within a (family, device) group matches the existing display lists
# (MEEG_CLASSIC_DISPLAY, FMRI_CLASSIC_DISPLAY, MEEG_FM_DISPLAY); downstream
# code derives those lists by filtering this iteration order.
# ---------------------------------------------------------------------------

MODELS: list[ModelEntry] = [
    # Task-specific EEG/MEG models, year-sorted.
    ModelEntry(
        name="ShallowFBCSPNet",
        family="classic",
        device="eeg",
        year=2017,
        bibtex="schirrmeister2017deep",
        config_name="ShallowFBCSPNet",
        cli_name="shallow_fbcsp_net",
        builder=_make_classic_braindecode_builder("ShallowFBCSPNet"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="Deep4Net",
        family="classic",
        device="eeg",
        year=2017,
        bibtex="schirrmeister2017deep",
        config_name="Deep4Net",
        cli_name="deep4net",
        builder=_make_classic_braindecode_builder("Deep4Net"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="EEGNet",
        family="classic",
        device="eeg",
        year=2018,
        bibtex="lawhern2018eegnet",
        config_name="EEGNet",
        cli_name="eegnet",
        builder=_make_classic_braindecode_builder("EEGNet"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="BDTCN",
        family="classic",
        device="eeg",
        year=2020,
        bibtex="gemein2020bdtcn",
        config_name="BDTCN",
        cli_name="bdtcn",
        builder=_make_classic_braindecode_builder("BDTCN"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="EEGConformer",
        family="classic",
        device="eeg",
        year=2022,
        bibtex="song2022eegconformer",
        config_name="EEGConformer",
        cli_name="eegconformer",
        builder=_make_classic_braindecode_builder("EEGConformer"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="ATCNet",
        family="classic",
        device="eeg",
        year=2022,
        bibtex="altaheri2022atcnet",
        config_name="ATCNet",
        cli_name="atcnet",
        builder=_make_classic_braindecode_builder("ATCNet"),
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="SimpleConvTimeAgg",
        family="classic",
        device="eeg",
        year=2023,
        bibtex="elouahidi2023eegsimpleconv",
        config_name="SimpleConvTimeAgg",
        cli_name="simpleconv_time_agg",
        builder=_build_simpleconv_time_agg,
        backbone_subtract=("output_head",),
    ),
    ModelEntry(
        name="CTNet",
        family="classic",
        device="eeg",
        year=2024,
        bibtex="zhao2024ctnet",
        config_name="CTNet",
        cli_name="ctnet",
        builder=_make_classic_braindecode_builder("CTNet"),
        backbone_subtract=("final_layer",),
    ),
    # fMRI baselines (kept in the registry to feed FMRI_CLASSIC_DISPLAY and
    # MODEL_DISPLAY_NAMES; they have no parameter count or year).
    ModelEntry(
        name="FmriLinear",
        family="classic",
        device="fmri",
        config_name="FmriLinear",
        cli_name="fmri_linear",
    ),
    ModelEntry(
        name="FmriMlp",
        family="classic",
        device="fmri",
        config_name="FmriMlp",
        cli_name="fmri_mlp",
    ),
    # Foundation models, year-sorted.
    ModelEntry(
        name="BENDR",
        family="foundation",
        device="eeg",
        year=2021,
        bibtex="kostas2021bendr",
        config_name="NtBendr",
        cli_name="bendr",
        variant="---",
        pretrain_strategy="Contrastive (wav2vec 2.0-style)",
        pretrain_data="TUEG",
        pretraining_overlap=_PT_BENDR,
        builder=_build_bendr,
        backbone_subtract=(),
    ),
    ModelEntry(
        name="BIOT",
        family="foundation",
        device="eeg",
        year=2023,
        bibtex="yang2023biot",
        config_name="BIOT",
        cli_name="biot",
        variant="---",
        pretrain_strategy="Contrastive + reconstruction",
        pretrain_data="6 datasets",
        pretraining_overlap=_PT_BIOT,
        builder=_build_biot,
        backbone_subtract=("final_layer", "classifier", "head"),
    ),
    ModelEntry(
        name="LaBraM",
        family="foundation",
        device="eeg",
        year=2024,
        bibtex="jiang2024labram",
        config_name="NtLabram",
        cli_name="labram",
        variant="Base",
        pretrain_strategy="VQ-VAE masked prediction",
        pretrain_data=r"16 datasets, ${\sim}$2.5K h",
        pretraining_overlap=_PT_LABRAM,
        builder=_build_labram,
        backbone_subtract=(),
    ),
    ModelEntry(
        name="CBraMod",
        family="foundation",
        device="eeg",
        year=2025,
        bibtex="wang2025cbramod",
        config_name="CBraMod",
        cli_name="cbramod",
        variant="---",
        pretrain_strategy="Masked patch reconstruction",
        pretrain_data="TUEG",
        pretraining_overlap=_PT_CBRAMOD,
        builder=_build_cbramod,
        backbone_subtract=("final_layer", "classifier", "head"),
    ),
    ModelEntry(
        name="LUNA",
        family="foundation",
        device="eeg",
        year=2025,
        bibtex="doner2025luna",
        config_name="NtLunaLarge",
        cli_name="luna",
        variant="Large",
        pretrain_strategy="Contrastive + MAE",
        pretrain_data="TUEG + Siena",
        pretraining_overlap=_PT_LUNA,
        builder=_build_luna,
        backbone_subtract=("final_layer",),
    ),
    ModelEntry(
        name="REVE",
        family="foundation",
        device="eeg",
        year=2025,
        bibtex="elouahidi2025reve",
        config_name="NtReve",
        cli_name="reve",
        variant="Base",
        pretrain_strategy="Masked autoencoding",
        pretrain_data=r"92 datasets, ${\sim}$60K h",
        pretraining_overlap=_PT_REVE,
        builder=_build_reve,
        backbone_subtract=("final_layer",),
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def lookup_by_name(name: str) -> ModelEntry | None:
    """Return the registry entry whose display ``name`` matches *name*, else
    ``None``.  Useful for replacing ``name in set(CLASSIC_DISPLAY)``-style
    membership checks with a direct registry lookup."""
    return _BY_NAME.get(name)


def lookup_by_cli_name(name: str) -> ModelEntry | None:
    """Return the registry entry whose ``cli_name`` matches *name*, else
    ``None``."""
    return _BY_CLI.get(name)


def lookup_by_config_name(name: str) -> ModelEntry | None:
    """Return the registry entry whose ``config_name`` matches *name*, else
    ``None``."""
    return _BY_CONFIG.get(name)


_BY_NAME: dict[str, ModelEntry] = {m.name: m for m in MODELS}
_BY_CLI: dict[str, ModelEntry] = {m.cli_name: m for m in MODELS if m.cli_name is not None}
_BY_CONFIG: dict[str, ModelEntry] = {
    m.config_name: m for m in MODELS if m.config_name is not None
}


# ---------------------------------------------------------------------------
# Parameter-count cache (round-trip with count_model_params.py)
# ---------------------------------------------------------------------------

PARAM_COUNT_CACHE_PATH: Path = Path(__file__).parent / "model_params.json"


def load_param_counts() -> dict[str, int]:
    """Load the cached backbone parameter counts produced by
    ``count_model_params.py``.  Missing file -> empty dict."""
    if not PARAM_COUNT_CACHE_PATH.exists():
        return {}
    with PARAM_COUNT_CACHE_PATH.open() as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def write_param_counts(counts: dict[str, int]) -> None:
    """Write the given backbone parameter counts to the JSON cache.

    Keys are written in :data:`MODELS` iteration order so the file is stable
    across runs regardless of dict insertion order at the call site.
    """
    name_order = {m.name: i for i, m in enumerate(MODELS)}
    ordered = dict(
        sorted(
            counts.items(),
            key=lambda kv: (name_order.get(kv[0], len(name_order)), kv[0]),
        )
    )
    with PARAM_COUNT_CACHE_PATH.open("w") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
