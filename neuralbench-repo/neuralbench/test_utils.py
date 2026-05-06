# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torch import nn

from neuraltrain.models.common import ChannelMerger, FourierEmb
from neuraltrain.models.preprocessor import OnTheFlyPreprocessor

from .modules import ChannelProjection, DownstreamWrapper, DownstreamWrapperModel


def test_downstream_wrapper():
    B, F, Fp = 8, 10, 3
    model = nn.Linear(F, 4)
    dummy_batch = {"input": torch.Tensor(B, F)}
    model = DownstreamWrapper().build(model, dummy_batch, Fp)

    assert isinstance(model, DownstreamWrapperModel)
    out = model(**dummy_batch)
    assert out.shape == (B, Fp)


class LinearOutDict(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.linear(x)
        return {
            "key1": out,
            "key2": out * 2,
        }


def test_downstream_wrapper_with_key():
    model = LinearOutDict(10, 4)
    B, F, Fp = 8, 10, 3
    dummy_batch = {"x": torch.Tensor(B, F)}
    model = DownstreamWrapper(
        model_output_key="key2",
    ).build(model, dummy_batch, Fp)

    assert isinstance(model, DownstreamWrapperModel)
    out = model(**dummy_batch)
    assert out.shape == (B, Fp)


def test_downstream_wrapper_with_preprocessor():
    """Test that on-the-fly preprocessing is applied inside the wrapper."""
    B, C, T, Fp = 4, 18, 200, 3
    model = nn.Identity()
    raw_input = torch.randn(B, C, T) * 1000.0
    dummy_batch = {"input": raw_input.clone()}

    preprocessor_config = OnTheFlyPreprocessor(
        scaler="StandardScaler",
        scale_dim=-1,
    )
    wrapper = DownstreamWrapper(
        on_the_fly_preprocessor=preprocessor_config,
        aggregation="flatten",
    )
    wrapped_model = wrapper.build(model, dummy_batch, Fp)

    assert isinstance(wrapped_model, DownstreamWrapperModel)
    assert wrapped_model.preprocessor is not None
    out = wrapped_model(**{"input": raw_input.clone()})
    assert out.shape == (B, Fp)

    # Verify that the preprocessing was actually applied: the output should
    # differ from a wrapper without preprocessing on the same input.
    wrapper_no_preproc = DownstreamWrapper(aggregation="flatten")
    model_no_preproc = wrapper_no_preproc.build(nn.Identity(), dummy_batch, Fp)
    out_no_preproc = model_no_preproc(**{"input": raw_input.clone()})
    msg = "Preprocessor should have modified the data going through the wrapper"
    assert not torch.allclose(out, out_no_preproc), msg


def test_downstream_wrapper_with_channel_merger():
    """Test that ChannelMerger projects channels before the inner model."""
    B, C_in, T = 4, 32, 200
    C_virtual = 8

    model = nn.Identity()
    channel_positions = torch.rand(B, C_in, 3)
    subject_ids = torch.zeros(B, dtype=torch.long)
    raw_input = torch.randn(B, C_in, T)

    dummy_batch = {
        "input": raw_input.clone(),
        "channel_positions": channel_positions,
        "subject_ids": subject_ids,
    }

    merger_config = ChannelMerger(
        n_virtual_channels=C_virtual,
        per_subject=False,
        fourier_emb_config=FourierEmb(n_dims=3),
    )
    wrapper = DownstreamWrapper(
        channel_adapter_config=merger_config,
        aggregation="flatten",
        probe_config=None,
    )
    n_outputs = C_virtual * T
    wrapped_model = wrapper.build(model, dummy_batch, n_outputs)

    assert isinstance(wrapped_model, DownstreamWrapperModel)
    assert wrapped_model.channel_adapter is not None
    assert wrapped_model._adapter_needs_positions

    out = wrapped_model(
        **{
            "input": raw_input.clone(),
            "channel_positions": channel_positions,
            "subject_ids": subject_ids,
        }
    )
    assert out.shape == (B, C_virtual * T)


def test_downstream_wrapper_with_channel_projection():
    """Test that ChannelProjection projects channels via Conv1d before the inner model."""
    B, C_in, T = 4, 32, 200
    C_target = 18

    model = nn.Identity()
    raw_input = torch.randn(B, C_in, T)
    dummy_batch = {"input": raw_input.clone()}

    proj_config = ChannelProjection(n_target_channels=C_target, max_norm=1.0)
    wrapper = DownstreamWrapper(
        channel_adapter_config=proj_config,
        aggregation="flatten",
        probe_config=None,
    )
    n_outputs = C_target * T
    wrapped_model = wrapper.build(model, dummy_batch, n_outputs)

    assert isinstance(wrapped_model, DownstreamWrapperModel)
    assert wrapped_model.channel_adapter is not None
    assert not wrapped_model._adapter_needs_positions

    out = wrapped_model(**{"input": raw_input.clone()})
    assert out.shape == (B, C_target * T)


def _identity_expected(
    target: list[str],
    inputs: list[str],
    rename: dict[str, str] | None = None,
) -> torch.Tensor:
    """Compute the one-hot weight pattern the identity init should produce."""
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
        pytest.param(
            ["A", "B", "C"],
            ["A", "B", "C"],
            None,
            1.0,
            id="with_max_norm",
        ),
    ],
)
def test_channel_projection_identity_patterns(target, inputs, rename, max_norm):
    """Identity init sets the expected one-hot pattern and zeros the bias.

    Covers exact-match, reorder+zero-pad-missing, rename/case-insensitive
    canonicalisation, and the ``Conv1dWithConstraint`` parametrize hook.
    A row for a covered target exposes its matched input through the
    forward pass; missing targets emit zeros.
    """
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

    x = torch.randn(2, len(inputs), 10)
    out = conv(x)
    for i in range(len(target)):
        src = expected[i, :, 0].nonzero(as_tuple=False).squeeze(-1).tolist()
        if src:
            assert torch.allclose(out[:, i, :], x[:, src[0], :])
        else:
            assert torch.allclose(out[:, i, :], torch.zeros_like(out[:, i, :]))


@pytest.mark.parametrize(
    "kwargs,build_kwargs,match",
    [
        pytest.param(
            {"n_target_channels": 2, "target_channel_names": ["A", "B"]},
            {"n_in_channels": 2},
            "input_channel_names",
            id="missing_input_names_at_build",
        ),
        pytest.param(
            {"n_target_channels": 3, "target_channel_names": ["A", "B"]},
            None,
            "target_channel_names",
            id="wrong_target_length_in_config",
        ),
    ],
)
def test_channel_projection_identity_validation(kwargs, build_kwargs, match):
    """``init='identity'`` rejects incomplete / inconsistent configurations."""
    if build_kwargs is None:
        with pytest.raises(ValueError, match=match):
            ChannelProjection(init="identity", max_norm=None, **kwargs)
    else:
        proj = ChannelProjection(init="identity", max_norm=None, **kwargs)
        with pytest.raises(ValueError, match=match):
            proj.build(**build_kwargs)


def test_channel_projection_identity_end_to_end():
    """Identity-init adapter is a pass-through inside a DownstreamWrapper."""
    target = ["Fp1", "Fp2", "Cz", "O1"]
    B, T = 3, 20
    raw_input = torch.randn(B, len(target), T)
    dummy_batch = {"input": raw_input.clone()}

    proj_config = ChannelProjection(
        n_target_channels=len(target),
        init="identity",
        target_channel_names=target,
        max_norm=None,
    )
    wrapper = DownstreamWrapper(
        channel_adapter_config=proj_config,
        aggregation="flatten",
        probe_config=None,
    )
    wrapped_model = wrapper.build(
        nn.Identity(), dummy_batch, len(target) * T, input_channel_names=target
    )
    out = wrapped_model(**{"input": raw_input.clone()})
    assert torch.allclose(out, raw_input.flatten(1))


def _bipolar_expected(
    target: list[str],
    inputs: list[str],
    rename: dict[str, str] | None = None,
) -> torch.Tensor:
    """Compute the +1/-1 pattern the bipolar init should *add* to Kaiming."""
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
            ["FP1-F7", "F7-T7"],
            ["UNK_0", "UNK_1", "UNK_2"],
            None,
            id="no_match",
        ),
        pytest.param(
            ["Fp1", "FP1-F7"],
            ["Fp1", "F7"],
            None,
            id="unipolar_and_bipolar_mix",
        ),
    ],
)
def test_channel_projection_bipolar_patterns(target, inputs, rename):
    """Bipolar init adds the +1/-1 pattern on top of the Kaiming baseline.

    Verified by building the same-shape adapter with ``init='random'`` under
    the same torch seed and asserting ``W_bipolar - W_random == pattern``.
    Fully missing rows stay at the Kaiming baseline so the forward output
    (and therefore the gradient through ``|STFT|``) is non-zero.
    """
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
    random = ChannelProjection(init="random", **kwargs).build(n_in_channels=len(inputs))

    expected = _bipolar_expected(target, inputs, rename)
    assert torch.allclose(bipolar.weight - random.weight, expected, atol=1e-6)
    assert bipolar.bias is not None
    assert torch.allclose(bipolar.bias, torch.zeros_like(bipolar.bias))

    # Rows left at the Kaiming baseline (partial / fully missing) must stay
    # non-zero -- otherwise BIOT's |STFT(0)| would freeze their gradient.
    pattern_row_abs = expected.abs().sum(dim=(1, 2))
    for i in range(len(target)):
        if pattern_row_abs[i] == 0:
            assert bipolar.weight[i].abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs,build_kwargs,match",
    [
        pytest.param(
            {"n_target_channels": 2, "target_channel_names": ["A-B", "C-D"]},
            {"n_in_channels": 4},
            "input_channel_names",
            id="missing_input_names_at_build",
        ),
        pytest.param(
            {"n_target_channels": 3, "target_channel_names": ["A-B", "C-D"]},
            None,
            "target_channel_names",
            id="wrong_target_length_in_config",
        ),
    ],
)
def test_channel_projection_bipolar_validation(kwargs, build_kwargs, match):
    """``init='bipolar'`` rejects incomplete / inconsistent configurations."""
    if build_kwargs is None:
        with pytest.raises(ValueError, match=match):
            ChannelProjection(init="bipolar", max_norm=None, **kwargs)
    else:
        proj = ChannelProjection(init="bipolar", max_norm=None, **kwargs)
        with pytest.raises(ValueError, match=match):
            proj.build(**build_kwargs)


def test_channel_projection_bipolar_end_to_end():
    """Bipolar-init adapter computes (A - B) for covered pairs end-to-end."""
    target = ["FP1-F7", "F7-T7"]
    inputs = ["Fp1", "F7", "T7"]
    B, T = 3, 20
    raw_input = torch.randn(B, len(inputs), T)
    dummy_batch = {"input": raw_input.clone()}

    proj_config = ChannelProjection(
        n_target_channels=len(target),
        init="bipolar",
        target_channel_names=target,
        max_norm=None,
    )
    wrapper = DownstreamWrapper(
        channel_adapter_config=proj_config,
        aggregation="flatten",
        probe_config=None,
    )
    wrapped_model = wrapper.build(
        nn.Identity(), dummy_batch, len(target) * T, input_channel_names=inputs
    )
    adapter = wrapped_model.channel_adapter
    assert adapter is not None
    # Strip the Kaiming baseline to isolate the bipolar pattern's contribution,
    # then verify the adapter computes (A - B) for each covered pair.
    with torch.no_grad():
        adapter.weight.copy_(_bipolar_expected(target, inputs))
    out = wrapped_model(**{"input": raw_input.clone()}).reshape(B, len(target), T)
    assert torch.allclose(out[:, 0, :], raw_input[:, 0, :] - raw_input[:, 1, :])
    assert torch.allclose(out[:, 1, :], raw_input[:, 1, :] - raw_input[:, 2, :])
