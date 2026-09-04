# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.modules`.

Covers ``DownstreamWrapper`` / ``DownstreamWrapperModel`` (aggregation, probe
head, probe_layer capture, preprocessor / channel-adapter wiring, dict-output
routing, LoRA injection) and ``ChannelProjection`` (identity / bipolar inits).
"""

import pytest
import torch
from torch import nn

from neuraltrain.models.common import ChannelMerger, FourierEmb
from neuraltrain.models.preprocessor import OnTheFlyPreprocessor

from .modules import (
    ChannelProjection,
    DownstreamWrapper,
    DownstreamWrapperModel,
    LoraConfig,
)

# ---------------------------------------------------------------------------
# DownstreamWrapper -- basic + probe_layer
# ---------------------------------------------------------------------------


def test_downstream_wrapper():
    B, F, Fp = 8, 10, 3
    dummy_batch = {"input": torch.randn(B, F)}
    wrapped = DownstreamWrapper().build(nn.Linear(F, 4), dummy_batch, Fp)
    assert isinstance(wrapped, DownstreamWrapperModel)
    assert wrapped(**dummy_batch).shape == (B, Fp)


def test_downstream_wrapper_probe_layer():
    B, F, Fp = 8, 10, 3
    model = nn.Sequential(nn.Linear(F, 16), nn.Linear(16, 4))
    dummy_batch = {"input": torch.Tensor(B, F)}
    wrapped = DownstreamWrapper(probe_layer="0").build(model, dummy_batch, Fp)
    # Probe is sized from layer "0"'s output (16), not the final layer (4).
    assert wrapped.probe.in_features == 16
    assert wrapped(**dummy_batch).shape == (B, Fp)


def test_downstream_wrapper_probe_layer_invalid():
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 4))
    with pytest.raises(AttributeError, match="not in Sequential"):
        DownstreamWrapper(probe_layer="no_such_layer").build(
            model, {"input": torch.Tensor(2, 10)}, 3
        )


def test_downstream_wrapper_probe_layer_requires_no_output_key():
    with pytest.raises(ValueError, match="model_output_key"):
        DownstreamWrapper(probe_layer="0", model_output_key="logits")


def test_downstream_wrapper_probe_batch_dim_requires_probe_layer():
    with pytest.raises(ValueError, match="probe_batch_dim only applies"):
        DownstreamWrapper(probe_batch_dim=1)


def test_downstream_wrapper_probe_layer_rejects_tuple_capture():
    # nn.RNN returns (output, h_n); probing a non-tensor capture must raise.
    model = nn.RNN(input_size=10, hidden_size=8, batch_first=True)
    with pytest.raises(TypeError, match="tensor-returning"):
        DownstreamWrapper(probe_layer="").build(
            model, {"input": torch.Tensor(2, 5, 10)}, 3
        )


class _SeqFirstEnc(nn.Module):
    """Emits sequence-first (T, B, D), like a ``batch_first=False`` transformer."""

    def __init__(self, n_in: int, emb: int):
        super().__init__()
        self.lin = nn.Linear(n_in, emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).transpose(0, 1)  # (B, T, F) -> (T, B, D)


class _SeqFirstProbeNet(nn.Module):
    """Probed submodule ``enc`` emits sequence-first (T, B, D)."""

    def __init__(self, n_in: int, emb: int):
        super().__init__()
        self.enc = _SeqFirstEnc(n_in, emb)
        self.head = nn.Linear(emb, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x).mean(0))


@pytest.mark.parametrize("aggregation", ["mean", "flatten", "first"])
def test_downstream_wrapper_probe_layer_seq_first(aggregation):
    # A (T, B, D) capture is auto-detected and moved to batch-first, so the
    # standard aggregations apply with no special-casing.
    B, T, F, D, Fp = 8, 5, 10, 6, 3
    wrapped = DownstreamWrapper(probe_layer="enc", aggregation=aggregation).build(
        _SeqFirstProbeNet(F, D), {"x": torch.Tensor(B, T, F)}, Fp
    )
    assert (
        wrapped.probe.in_features
        == {"mean": D, "flatten": T * D, "first": D}[aggregation]
    )
    assert wrapped(x=torch.Tensor(B, T, F)).shape == (B, Fp)


def test_downstream_wrapper_probe_layer_batch_seq_collision():
    # Two-pass detection resolves the batch axis even when batch == seq length.
    n, F, D, Fp = 5, 10, 6, 3
    wrapped = DownstreamWrapper(probe_layer="enc", aggregation="mean").build(
        _SeqFirstProbeNet(F, D), {"x": torch.Tensor(n, n, F)}, Fp
    )
    assert wrapped.probe.in_features == D
    assert wrapped(x=torch.Tensor(n, n, F)).shape == (n, Fp)


def test_downstream_wrapper_probe_layer_batch_dim_override():
    # Explicit probe_batch_dim=1 skips auto-detection for the (T, B, D) capture.
    B, T, F, D, Fp = 8, 5, 10, 6, 3
    wrapped = DownstreamWrapper(
        probe_layer="enc", aggregation="mean", probe_batch_dim=1
    ).build(_SeqFirstProbeNet(F, D), {"x": torch.Tensor(B, T, F)}, Fp)
    assert wrapped.probe.in_features == D
    assert wrapped(x=torch.Tensor(B, T, F)).shape == (B, Fp)


# ---------------------------------------------------------------------------
# Probe head
# ---------------------------------------------------------------------------


class _TokenModel2D(nn.Module):
    """Returns 2-D per-sample token embeddings ``(B, n_patches, emb)``."""

    def __init__(self, n_in: int, n_patches: int, emb: int):
        super().__init__()
        self.n_patches, self.emb = n_patches, emb
        self.lin = nn.Linear(n_in, n_patches * emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).reshape(x.shape[0], self.n_patches, self.emb)


class _TokenModel3D(nn.Module):
    """Returns 3-D per-sample token embeddings ``(B, n_chans, n_patches, emb)``."""

    def __init__(self, n_in: int, n_chans: int, n_patches: int, emb: int):
        super().__init__()
        self.shape = (n_chans, n_patches, emb)
        self.lin = nn.Linear(n_in, n_chans * n_patches * emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).reshape(x.shape[0], *self.shape)


@pytest.mark.parametrize("ndim", [2, 3])
def test_attention_probe_is_trainable_on_frozen_backbone(ndim):
    B, F, C, P, E, Fp = 8, 10, 3, 4, 5, 3
    model = _TokenModel2D(F, P, E) if ndim == 2 else _TokenModel3D(F, C, P, E)
    dummy = {"x": torch.randn(B, F)}
    wrapped = DownstreamWrapper(
        aggregation=None, probe_config="attention", layers_to_unfreeze=[""]
    ).build(model, dummy, Fp)
    probe_params = list(wrapped.probe.parameters())
    assert not any(p.requires_grad for p in wrapped.wrapped_model.parameters())
    assert probe_params and all(p.requires_grad for p in probe_params)
    assert wrapped(**dummy).shape == (B, Fp)


def test_attention_probe_requires_no_aggregation():
    with pytest.raises(ValueError, match="requires aggregation=None"):
        DownstreamWrapper(aggregation="mean", probe_config="attention")


# ---------------------------------------------------------------------------
# DownstreamWrapper -- output-key routing, preprocessor, channel adapters
# ---------------------------------------------------------------------------


class LinearOutDict(nn.Module):
    """Linear layer returning ``{"key1": Wx, "key2": 2 * Wx}`` to test key routing."""

    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.linear(x)
        return {"key1": out, "key2": out * 2}


def test_downstream_wrapper_routes_selected_dict_key():
    B, F, n_outputs = 8, 10, 4
    inner = LinearOutDict(F, n_outputs)
    dummy_batch = {"x": torch.randn(B, F)}

    out_key1 = DownstreamWrapper(
        model_output_key="key1", aggregation="flatten", probe_config=None
    ).build(inner, dummy_batch, n_outputs)(**dummy_batch)
    out_key2 = DownstreamWrapper(
        model_output_key="key2", aggregation="flatten", probe_config=None
    ).build(inner, dummy_batch, n_outputs)(**dummy_batch)

    assert out_key1.shape == (B, n_outputs)
    # key2 == 2*key1 pins routing to the value, not just the shape.
    assert torch.allclose(out_key2, 2 * out_key1)


def test_downstream_wrapper_with_preprocessor():
    B, C, T, Fp = 4, 18, 200, 3
    raw_input = torch.randn(B, C, T) * 1000.0
    dummy_batch = {"input": raw_input.clone()}

    wrapped = DownstreamWrapper(
        on_the_fly_preprocessor=OnTheFlyPreprocessor(
            scaler="StandardScaler", scale_dim=-1
        ),
        aggregation="flatten",
    ).build(nn.Identity(), dummy_batch, Fp)
    assert wrapped.preprocessor is not None
    out = wrapped(**{"input": raw_input.clone()})
    assert out.shape == (B, Fp)

    no_preproc = DownstreamWrapper(aggregation="flatten").build(
        nn.Identity(), dummy_batch, Fp
    )
    out_no_preproc = no_preproc(**{"input": raw_input.clone()})
    assert not torch.allclose(out, out_no_preproc), "preprocessor did not alter output"


def test_downstream_wrapper_with_channel_merger():
    B, C_in, T, C_virtual = 4, 32, 200, 8
    channel_positions = torch.rand(B, C_in, 3)
    subject_ids = torch.zeros(B, dtype=torch.long)
    raw_input = torch.randn(B, C_in, T)
    dummy_batch = {
        "input": raw_input.clone(),
        "channel_positions": channel_positions,
        "subject_ids": subject_ids,
    }

    wrapped = DownstreamWrapper(
        channel_adapter_config=ChannelMerger(
            n_virtual_channels=C_virtual,
            per_subject=False,
            fourier_emb_config=FourierEmb(n_dims=3),
        ),
        aggregation="flatten",
        probe_config=None,
    ).build(nn.Identity(), dummy_batch, C_virtual * T)
    assert wrapped.channel_adapter is not None
    assert wrapped._adapter_needs_positions

    out = wrapped(
        input=raw_input.clone(),
        channel_positions=channel_positions,
        subject_ids=subject_ids,
    )
    assert out.shape == (B, C_virtual * T)
    # The merger is position-conditioned: different positions -> different output.
    out_other = wrapped(
        input=raw_input.clone(),
        channel_positions=torch.rand(B, C_in, 3),
        subject_ids=subject_ids,
    )
    assert not torch.allclose(out, out_other)


def test_downstream_wrapper_with_channel_projection():
    B, C_in, T, C_target = 4, 32, 200, 18
    raw_input = torch.randn(B, C_in, T)
    dummy_batch = {"input": raw_input.clone()}

    wrapped = DownstreamWrapper(
        channel_adapter_config=ChannelProjection(
            n_target_channels=C_target, max_norm=1.0
        ),
        aggregation="flatten",
        probe_config=None,
    ).build(nn.Identity(), dummy_batch, C_target * T)
    assert wrapped.channel_adapter is not None
    assert not wrapped._adapter_needs_positions
    assert wrapped(**{"input": raw_input.clone()}).shape == (B, C_target * T)


# ---------------------------------------------------------------------------
# ChannelProjection -- identity / bipolar init
# ---------------------------------------------------------------------------


def _identity_expected(
    target: list[str],
    inputs: list[str],
    rename: dict[str, str] | None = None,
) -> torch.Tensor:
    """One-hot weight pattern the identity init should produce."""
    canon_inputs = [(rename or {}).get(n, n).upper() for n in inputs]
    target_upper = [t.upper() for t in target]
    weight = torch.zeros(len(target), len(inputs), 1)
    used_inputs: set[int] = set()
    for i, tu in enumerate(target_upper):
        for j, cn in enumerate(canon_inputs):
            if cn == tu and j not in used_inputs:
                weight[i, j, 0] = 1.0
                used_inputs.add(j)
                break
    return weight


@pytest.mark.parametrize(
    "target,inputs,rename,max_norm",
    [
        pytest.param(
            ["Fp1", "Fp2", "Cz", "O1"],
            ["Fp1", "Fp2", "Cz", "O1"],
            None,
            None,
            id="exact_match",
        ),
        pytest.param(
            ["Fp1", "Fp2", "Cz", "O1", "O2"],
            ["Cz", "Fp1", "EXTRA", "O1"],
            None,
            None,
            id="reorder_and_pad",
        ),
        pytest.param(
            ["T7", "P7", "FP2"],
            ["T3", "T5", "e9"],
            {"T3": "T7", "T5": "P7", "e9": "Fp2"},
            None,
            id="rename_and_case_insensitive",
        ),
        pytest.param(["A", "B", "C"], ["A", "B", "C"], None, 1.0, id="with_max_norm"),
    ],
)
def test_channel_projection_identity_patterns(target, inputs, rename, max_norm):
    proj = ChannelProjection(
        n_target_channels=len(target),
        init="identity",
        target_channel_names=target,
        rename_mapping=rename,
        max_norm=max_norm,
    )
    conv = proj.build(n_in_channels=len(inputs), input_channel_names=inputs)

    expected = _identity_expected(target, inputs, rename)
    assert torch.allclose(conv.weight, expected)
    assert conv.bias is not None
    assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    # A covered target row exposes its matched input; a missing target emits zeros.
    x = torch.randn(2, len(inputs), 10)
    out = conv(x)
    for i in range(len(target)):
        src = expected[i, :, 0].nonzero(as_tuple=False).squeeze(-1).tolist()
        if src:
            assert torch.allclose(out[:, i, :], x[:, src[0], :])
        else:
            assert torch.allclose(out[:, i, :], torch.zeros_like(out[:, i, :]))


def _bipolar_expected(
    target: list[str],
    inputs: list[str],
    rename: dict[str, str] | None = None,
) -> torch.Tensor:
    """+1/-1 pattern the bipolar init should *add* to the Kaiming baseline."""
    canon_inputs = [(rename or {}).get(n, n).upper() for n in inputs]
    canon_to_idx: dict[str, int] = {}
    for j, cn in enumerate(canon_inputs):
        canon_to_idx.setdefault(cn, j)
    pattern = torch.zeros(len(target), len(inputs), 1)
    for i, tname in enumerate(target):
        if "-" in tname:
            pos, neg = tname.split("-", 1)
        else:
            pos, neg = tname, None
        pos_idx = canon_to_idx.get(pos.upper())
        neg_idx = canon_to_idx.get(neg.upper()) if neg else None
        if neg is None:
            if pos_idx is not None:
                pattern[i, pos_idx, 0] = 1.0
        elif pos_idx is not None and neg_idx is not None:
            pattern[i, pos_idx, 0] = 1.0
            pattern[i, neg_idx, 0] = -1.0
        # Partial / fully-missing rows stay at 0 (additive means row = Kaiming).
    return pattern


@pytest.mark.parametrize(
    "target,inputs,rename",
    [
        pytest.param(
            ["FP1-F7", "F7-T7", "C3-A2"],
            ["Fp1", "F7", "T7", "C3"],
            None,
            id="full_and_partial_mix",
        ),
        pytest.param(
            ["FP1-F7", "T7-P7"],
            ["Fp1", "T3", "T5"],
            {"T3": "T7", "T5": "P7"},
            id="rename",
        ),
        pytest.param(
            ["FP1-F7", "F7-T7"], ["UNK_0", "UNK_1", "UNK_2"], None, id="no_match"
        ),
        pytest.param(
            ["Fp1", "FP1-F7"], ["Fp1", "F7"], None, id="unipolar_and_bipolar_mix"
        ),
    ],
)
def test_channel_projection_bipolar_patterns(target, inputs, rename):
    # Build the same-shape adapter with init='random' under the same seed and
    # assert W_bipolar - W_kaiming == pattern.
    kwargs: dict = dict(
        n_target_channels=len(target),
        target_channel_names=target,
        rename_mapping=rename,
        max_norm=None,
    )
    torch.manual_seed(0)
    bipolar = ChannelProjection(init="bipolar", **kwargs).build(
        n_in_channels=len(inputs), input_channel_names=inputs
    )
    torch.manual_seed(0)
    kaiming = ChannelProjection(init="random", **kwargs).build(n_in_channels=len(inputs))

    expected = _bipolar_expected(target, inputs, rename)
    assert torch.allclose(bipolar.weight - kaiming.weight, expected, atol=1e-6)
    assert bipolar.bias is not None
    assert torch.allclose(bipolar.bias, torch.zeros_like(bipolar.bias))

    # Rows left at the Kaiming baseline must stay non-zero, else BIOT's
    # |STFT(0)| would freeze their gradient.
    pattern_row_abs = expected.abs().sum(dim=(1, 2))
    for i in range(len(target)):
        if pattern_row_abs[i] == 0:
            assert bipolar.weight[i].abs().sum() > 0


@pytest.mark.parametrize(
    "init,target_names",
    [("identity", ["A", "B"]), ("bipolar", ["A-B", "C-D"])],
)
def test_channel_projection_rejects_wrong_target_length(init, target_names):
    with pytest.raises(ValueError, match="target_channel_names"):
        ChannelProjection(
            init=init,
            max_norm=None,
            n_target_channels=3,
            target_channel_names=target_names,
        )


@pytest.mark.parametrize(
    "init,target_names",
    [("identity", ["A", "B"]), ("bipolar", ["A-B", "C-D"])],
)
def test_channel_projection_requires_input_names_at_build(init, target_names):
    proj = ChannelProjection(
        init=init,
        max_norm=None,
        n_target_channels=len(target_names),
        target_channel_names=target_names,
    )
    with pytest.raises(ValueError, match="input_channel_names"):
        proj.build(n_in_channels=2)


def test_channel_projection_identity_end_to_end():
    # Identity-init adapter is a pass-through inside a DownstreamWrapper.
    target = ["Fp1", "Fp2", "Cz", "O1"]
    B, T = 3, 20
    raw_input = torch.randn(B, len(target), T)
    dummy_batch = {"input": raw_input.clone()}

    wrapped = DownstreamWrapper(
        channel_adapter_config=ChannelProjection(
            n_target_channels=len(target),
            init="identity",
            target_channel_names=target,
            max_norm=None,
        ),
        aggregation="flatten",
        probe_config=None,
    ).build(nn.Identity(), dummy_batch, len(target) * T, input_channel_names=target)
    out = wrapped(**{"input": raw_input.clone()})
    assert torch.allclose(out, raw_input.flatten(1))


def test_channel_projection_bipolar_end_to_end():
    # Bipolar-init adapter computes (A - B) for covered pairs end-to-end.
    target = ["FP1-F7", "F7-T7"]
    inputs = ["Fp1", "F7", "T7"]
    B, T = 3, 20
    raw_input = torch.randn(B, len(inputs), T)
    dummy_batch = {"input": raw_input.clone()}

    wrapped = DownstreamWrapper(
        channel_adapter_config=ChannelProjection(
            n_target_channels=len(target),
            init="bipolar",
            target_channel_names=target,
            max_norm=None,
        ),
        aggregation="flatten",
        probe_config=None,
    ).build(nn.Identity(), dummy_batch, len(target) * T, input_channel_names=inputs)
    adapter = wrapped.channel_adapter
    assert adapter is not None
    # Strip the Kaiming baseline to isolate the bipolar pattern's contribution.
    with torch.no_grad():
        adapter.weight.copy_(_bipolar_expected(target, inputs))
    out = wrapped(**{"input": raw_input.clone()}).reshape(B, len(target), T)
    assert torch.allclose(out[:, 0, :], raw_input[:, 0, :] - raw_input[:, 1, :])
    assert torch.allclose(out[:, 1, :], raw_input[:, 1, :] - raw_input[:, 2, :])


# ---------------------------------------------------------------------------
# DownstreamWrapper -- LoRA injection
# ---------------------------------------------------------------------------


class _TinyAttn(nn.Module):
    """Toy attention block with named Q/K/V/O ``nn.Linear`` leaves.

    PEFT matches ``target_modules`` on leaf names, so only the names matter.
    """

    def __init__(self, d: int = 16):
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))


_QKVO = ["to_q", "to_k", "to_v", "to_out"]


def _lora_wrapper(*, targets, r=4, target_modules=None) -> DownstreamWrapper:
    """A frozen-backbone LoRA wrapper; only the target lists vary between tests."""
    return DownstreamWrapper(
        layers_to_unfreeze=[""],
        aggregation="mean",
        probe_config="linear",
        lora_config=LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=target_modules
        ),
        lora_target_modules=targets,
    )


def _lora_leaf_names(wrapped) -> set[str]:
    """Leaf-module names that received a LoRA adapter."""
    parents = {
        name.split(".lora_")[0]
        for name, _ in wrapped.wrapped_model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    return {p.rsplit(".", 1)[-1] for p in parents}


def _lora_params(wrapped) -> dict[str, torch.nn.Parameter]:
    return {
        n: p
        for n, p in wrapped.wrapped_model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }


def test_downstream_wrapper_lora_wraps_freezes_and_trains():
    B, d, n_out = 2, 16, 3
    dummy_batch = {"x": torch.randn(B, 5, d)}
    wrapped = _lora_wrapper(targets=_QKVO).build(_TinyAttn(d=d), dummy_batch, n_out)

    assert _lora_leaf_names(wrapped) == set(_QKVO)
    # inside wrapped_model only the adapters train; the probe sits outside
    inner_trainable = {
        n for n, p in wrapped.wrapped_model.named_parameters() if p.requires_grad
    }
    assert inner_trainable and all(
        ("lora_A" in n or "lora_B" in n) for n in inner_trainable
    )
    assert any(p.requires_grad for p in wrapped.probe.parameters())

    # 4 Linears of 16->16: per module 16r (A) + 16r (B); total 128r, so r reaches peft
    assert sum(p.numel() for p in _lora_params(wrapped).values()) == 128 * 4
    doubled = _lora_wrapper(targets=_QKVO, r=8).build(_TinyAttn(d=d), dummy_batch, n_out)
    assert sum(p.numel() for p in _lora_params(doubled).values()) == 128 * 8

    out = wrapped(**dummy_batch)
    assert out.shape == (B, n_out)

    # lora_B is zero-init, so a non-zero grad proves it is wired into autograd
    out.pow(2).mean().backward()
    grads = {n: p.grad for n, p in _lora_params(wrapped).items()}
    assert grads and all(g is not None for g in grads.values())
    assert all(g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n)


def test_downstream_wrapper_lora_target_module_resolution():
    # Missing targets raise; an explicit lora_config subset wins over the YAML list.
    dummy_batch = {"x": torch.randn(2, 5, 16)}

    with pytest.raises(ValueError, match="LoRA requires target_modules"):
        _lora_wrapper(targets=None).build(_TinyAttn(), dummy_batch, 3)

    override = _lora_wrapper(targets=_QKVO, target_modules=["to_q"])
    assert _lora_leaf_names(override.build(_TinyAttn(), dummy_batch, 3)) == {"to_q"}


class _QkvProjAttn(nn.Module):
    """Attention with ``qkv`` + ``proj`` linear leaves (the LoRA targets)."""

    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        return self.proj(self.qkv(x)[..., :d])


class _AttnWithDecoyProj(nn.Module):
    """``attn.qkv``/``attn.proj`` beside decoy ``proj`` leaves that must be skipped.

    Mirrors EEGPT, where a bare ``proj`` would also match a patch-embed Conv2d.
    """

    def __init__(self, d: int = 16):
        super().__init__()
        self.attn = _QkvProjAttn(d)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(1, 1, 1)
        self.tokenizer = nn.Module()
        self.tokenizer.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


def test_downstream_wrapper_lora_dotted_target_skips_decoy_proj():
    dummy_batch = {"x": torch.randn(2, 5, 16)}
    wrapped = _lora_wrapper(targets=["qkv", "attn.proj"]).build(
        _AttnWithDecoyProj(), dummy_batch, 3
    )

    lora_names = [n for n, _ in wrapped.wrapped_model.named_parameters() if "lora_" in n]
    assert any("attn.qkv." in n for n in lora_names)
    assert any("attn.proj." in n for n in lora_names)
    assert not any("patch_embed" in n for n in lora_names)
    assert not any("tokenizer" in n for n in lora_names)
