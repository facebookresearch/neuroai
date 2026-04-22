# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Common modules to be used with brain models."""

# pylint: disable=unused-variable

import math
import typing as tp
from collections import deque

import torch
from torch import nn
from torchvision.ops import MLP

from neuraltrain.models.base import BaseModelConfig

INVALID_POS_VALUE = -0.1  # See ns.extractors.ChannelPositions.INVALID_VALUE


def parse_bipolar_name(name: str) -> tuple[str, str] | None:
    """Split a bipolar channel name into its anode and cathode.

    Returns ``("Fp1", "F3")`` for ``"Fp1-F3"``, or ``None`` when *name*
    does not look like a bipolar pair (i.e. does not contain exactly one
    hyphen separating two non-empty parts).
    """
    parts = name.split("-")
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return None


def compute_temporal_adjustment(n_times: int, patch_size: int) -> tuple[int, int]:
    """Compute how to align *n_times* to a multiple of *patch_size*.

    Returns ``(pad_right, truncate_right)`` -- at most one is non-zero.

    * ``n_times < patch_size`` -- pad to one full patch.
    * ``n_times`` not divisible by ``patch_size`` -- truncate the remainder.
    * Already aligned -- ``(0, 0)``.
    """
    if n_times < patch_size:
        return patch_size - n_times, 0
    remainder = n_times % patch_size
    if remainder != 0:
        return 0, remainder
    return 0, 0


def apply_temporal_adjustment(
    x: torch.Tensor,
    pad_right: int = 0,
    truncate_right: int = 0,
) -> torch.Tensor:
    """Zero-pad or truncate the last (time) dimension of *x*."""
    if pad_right > 0:
        return torch.nn.functional.pad(x, (0, pad_right))
    if truncate_right > 0:
        return x[..., :-truncate_right]
    return x


class BahdanauAttention(nn.Module):
    """Bahdanau attention from [1]_.

    Implementation inspired from pytorch's seq2seq tutorial:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-decoder

    .. [1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by
           jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        if input_size is None:
            self.Wa = nn.LazyLinear(hidden_size)
            self.Ua = nn.LazyLinear(hidden_size)
        else:
            self.Wa = nn.Linear(input_size, hidden_size)
            self.Ua = nn.Linear(input_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, keys, queries=None):
        """
        Parameters
        ----------
        keys:
            Key tensor of shape (batch_size, n_features, n_times).
        queries:
            Optional query tensor of shape (batch_size, n_features, n_times).
            If None, only keys are used.
        """
        keys = keys.transpose(2, 1)  # (B, F, T) -> (B, T, F)
        sum_ = self.Wa(keys)
        if queries is not None:
            queries = queries.transpose(2, 1)
            assert queries.shape == keys.shape
            sum_ += self.Ua(queries)

        scores = self.Va(torch.tanh(sum_))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        context = context.transpose(2, 1)  # (B, 1, F) -> (B, F, 1)

        return context


class ChannelDropout(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("See brainmagick.models.common.")

    def forward(self, x):
        raise NotImplementedError


class SubjectLayers(BaseModelConfig):
    """Configuration for per-subject linear projections.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to allocate weight matrices for.
    bias : bool
        Include a bias term in each subject's projection.
    init_id : bool
        Initialize projection matrices to the identity (requires
        ``in_channels == out_channels``).
    mode : {"gather", "for_loop"}
        ``"gather"`` builds a ``(B, C_in, C_out)`` tensor via index_select
        (fast but memory-heavy for large channel counts).  ``"for_loop"``
        iterates over unique subjects (slower but lighter).
    subject_dropout : float or None
        Probability of replacing a subject index with a shared "dropout
        subject" during training.  Required when *average_subjects* is True.
    average_subjects : bool
        At inference time, use the shared dropout-subject weights for all
        examples.  Requires *subject_dropout* to be set.
    """

    n_subjects: int = 200
    bias: bool = True
    init_id: bool = False
    mode: tp.Literal["gather", "for_loop"] = "gather"
    subject_dropout: float | None = None
    average_subjects: bool = False

    def build(self, in_channels: int, out_channels: int) -> nn.Module:
        kwargs = self.model_dump()
        del kwargs["name"]
        return SubjectLayersModel(in_channels, out_channels, **kwargs)


class SubjectLayersModel(nn.Module):
    """Per subject linear projection.

    Parameters
    ----------
    in_channels :
        Number of input channels.
    out_channels :
        Number of output channels.
    n_subjects :
        Number of subjects to initialize weights for.
    bias:
        If True, use a bias term.
    init_id :
        If True, initialize the projection matrices with the identity.
    mode :
        How to apply the linear projection. With "gather" (original implementation), a tensor of
        shape (batch_size, in_channels, out_channels) containing the projection matrices for each
        example in the batch is first created. This tensor can be very large when the number of
        channels is high (e.g. when using on fMRI data with many input voxels). In this case, it
        may be better to use "for_loop": this will loop over each unique subject in the batch to
        apply the projection separately.
    subject_dropout :
        If not None, probability with which a subject's index is replaced by a shared "dropout
        subject" during training. An extra row is added to the weight matrix for this purpose.
        Required when ``average_subjects`` is True.
    average_subjects :
        If True, use the shared dropout-subject weights for all examples (inference shortcut).
        Requires ``subject_dropout`` to be set.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        n_subjects: int = 200,
        bias: bool = True,
        init_id: bool = False,
        mode: tp.Literal["gather", "for_loop"] = "gather",
        subject_dropout: float | None = None,
        average_subjects: bool = False,
    ):
        super().__init__()
        self.n_subjects = n_subjects
        self.subject_dropout = subject_dropout
        num_weight_subjects = n_subjects + 1 if subject_dropout else n_subjects
        self.weights = nn.Parameter(
            torch.empty(num_weight_subjects, in_channels, out_channels)
        )
        self.bias = (
            nn.Parameter(torch.empty(num_weight_subjects, out_channels)) if bias else None
        )
        if init_id:
            if in_channels != out_channels:
                raise ValueError(
                    "in_channels and out_channels must be the same for identity initialization."
                )
            self.weights.data[:] = torch.eye(in_channels)[None]
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            self.weights.data.normal_()
            if self.bias is not None:
                self.bias.data.normal_()
        self.weights.data *= 1 / in_channels**0.5
        if self.bias is not None:
            self.bias.data *= 1 / in_channels**0.5
        self.average_subjects = average_subjects
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, in_channels) or (batch_size, in_channels, n_times)
        subjects: torch.Tensor,  # (batch_size,)
    ) -> (
        torch.Tensor
    ):  # (batch_size, out_channels) or (batch_size, out_channels, n_times)
        if x.ndim not in [2, 3]:
            raise ValueError(f"Expected shape (B, C, T) or (B, C), got {x.shape}")
        if x.ndim == 2:
            has_time_dimension = False
            x = x.unsqueeze(2)
        else:
            has_time_dimension = True
        B, C, T = x.shape
        N, C, D = self.weights.shape

        if self.average_subjects:
            if not self.subject_dropout:
                raise ValueError("subject_dropout must be set to average subjects.")
            weights = self.weights[self.n_subjects]
            out = torch.einsum("bct,cd->bdt", x, weights)
            if self.bias is not None:
                out += self.bias[self.n_subjects].view(1, D, 1)
            return out if has_time_dimension else out.squeeze(2)
        else:
            if self.training and self.subject_dropout:
                subject_dropout_mask = (
                    torch.rand(subjects.shape, device=subjects.device)
                    < self.subject_dropout
                )
                subjects = subjects.clone()
                subjects[subject_dropout_mask] = self.n_subjects
            if subjects.max() >= N:
                raise ValueError(
                    f"Subject index {subjects.max()} out of range for {N} weight slots"
                )
            if self.mode == "gather":
                weights = self.weights.index_select(0, subjects.flatten())
                out = torch.einsum("bct,bcd->bdt", x, weights)
                if self.bias is not None:
                    out += self.bias.index_select(0, subjects.flatten()).view(B, D, 1)
            else:
                out = torch.empty((B, D, T), device=x.device)
                for subject in subjects.unique():
                    mask = subjects.reshape(-1) == subject
                    out[mask] = torch.einsum(
                        "bct,cd->bdt", x[mask], self.weights[subject]
                    )
                    if self.bias is not None:
                        out[mask] += self.bias[subject].view(1, D, 1)
        return out if has_time_dimension else out.squeeze(2)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayersModel({C}, {D}, {S})"


class FourierEmb(BaseModelConfig):
    """Configuration for Fourier positional embedding.

    Parameters
    ----------
    n_freqs :
        Number of frequencies (harmonics) used to encode **one** dimension.
    total_dim :
        If provided instead of `n_freqs`, this will be used to compute the number of
        frequencies following this relationship:

            n_freqs = (total_dim / 2) ** (1 / n_dims)

        If the resulting `n_freqs` is not an integer an exception will be raised.
    n_dims :
        Number of dimensions to embed. This should be 2 for 2D positions (e.g. MNE layouts) or 3
        for 3D positions (e.g. MNE montages).
    margin :
        How much to extend the range of the embedding to avoid edge effects.
    """

    n_freqs: int | None = 12
    total_dim: int | None = None
    n_dims: int = 2
    margin: float = 0.2

    def build(self) -> "FourierEmbModel":
        if self.total_dim is not None and self.n_freqs is None:
            n_freqs = (self.total_dim / 2) ** (1 / self.n_dims)
            if abs(n_freqs - round(n_freqs)) > 1e-6:  # Check if n_freqs is integer
                raise ValueError("(total_dim / 2) ** (1 / n_dims) must be an integer.")
            n_freqs = round(n_freqs)
        elif self.n_freqs is not None and self.total_dim is None:
            n_freqs = self.n_freqs
        else:
            raise ValueError("Exactly one of n_freqs and total_dim must be provided.")

        return FourierEmbModel(
            n_freqs=n_freqs,
            n_dims=self.n_dims,
            margin=self.margin,
        )


class FourierEmbModel(nn.Module):
    """Fourier positional embedding.

    Unlike traditional embedding this is not using exponential periods for cosines and sinuses, but
    typical `2 pi k` which can represent any function over [0, 1]. As this function would be
    necessarily periodic, we take a bit of margin and do over e.g. [-0.2, 1.2].
    """

    def __init__(
        self,
        n_freqs: int,
        n_dims: int,
        margin: float,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_dims = n_dims
        self.margin = margin

        # Precompute sin/cos arguments
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @property
    def total_dim(self) -> int:
        """Total dimension of the embedding."""
        return (self.n_freqs**self.n_dims) * 2

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        """Outer sum between the last dimensions of `x`.

        x.shape[-1] is expected to match the dimensions of the grid, between which the outer sum
        is computed. For example, if x.shape[-1] == 2, the outer sum will be computed between the
        last two dimensions of `x`, i.e. x[..., 0] and x[..., 1].
        """
        inds = deque([slice(None)] + [None] * (x.shape[-1] - 1))
        out = x[..., 0][(...,) + tuple(inds)]
        for i in range(1, x.shape[-1]):
            inds.rotate()
            out = out + x[..., i][(...,) + tuple(inds)]
        return out

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute Fourier embedding for the given positions.

        Parameters
        ----------
        positions : Tensor
            Coordinates of shape ``(..., D)`` where ``D`` equals *n_dims*.
        """
        *O, D = positions.shape
        if D != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} positions, got {D}.")
        positions = positions + self.margin
        locs = torch.einsum("bcd,f->bcfd", positions, self.pos)
        loc_grid = self._outer_sum(locs).view(*O, -1)
        emb = torch.cat(
            [
                torch.cos(loc_grid),
                torch.sin(loc_grid),
            ],
            dim=-1,
        )
        return emb


class ChannelMerger(BaseModelConfig):
    """Configuration for the ChannelMerger module.

    Parameters
    ----------
    embed_ref :
        Also embed the reference position, e.g. to enable handling bipolar channels. This requires
        passing both `positions` and `ref_positions` to `forward()`.
    dropout_around_channel :
        If True, randomly sample a channel to apply dropout around. If False, randomly sample a
        point in [0, 1] ^ D, where D is the number of dimensions (2 or 3), around which to apply
        dropout.
    unmerge :
        If True, unmerge (rather than merge) channels. This is useful to compute the inverse
        operation of a default `ChannelMerger`. In this case, the input to `forward()` should be of
        shape (B, n_virtual_channels, T).
    invalid_value :
        If all position dimensions for a channel are equal to `invalid_value`, the channel will be
        masked out. This is useful when examples within a batch contain different channels and
        therefore some channels need to be ignored for some of the examples.

        NOTE: `ns.extractors.ChannelPositions` defines this value as well.
    """

    n_virtual_channels: int = 270
    fourier_emb_config: FourierEmb = FourierEmb(
        n_freqs=None,
        total_dim=288,
        n_dims=2,
    )
    dropout: float = 0
    dropout_around_channel: bool = False
    usage_penalty: float = 0.0
    n_subjects: int = 200
    per_subject: bool = False
    embed_ref: bool = False
    unmerge: bool = False
    invalid_value: float = INVALID_POS_VALUE

    def build(self) -> "ChannelMergerModel":
        return ChannelMergerModel(self)


class ChannelMergerModel(nn.Module):
    """``nn.Module`` implementation of :class:`ChannelMerger`.

    Merges (or unmerges) channels via Fourier-embedding-based spatial attention.
    """

    def __init__(self, config: ChannelMerger = ChannelMerger()):
        super().__init__()
        self.embedding = config.fourier_emb_config.build()
        self.n_dims = self.embedding.n_dims
        pos_dim = self.embedding.total_dim
        assert isinstance(pos_dim, int)  # for mypy

        self.per_subject = config.per_subject
        self.embed_ref = config.embed_ref
        n_params_pos_dim = pos_dim * 2 if self.embed_ref else pos_dim
        if self.per_subject:
            self.heads = nn.Parameter(
                torch.randn(
                    config.n_subjects, config.n_virtual_channels, n_params_pos_dim
                )
            )
        else:
            self.heads = nn.Parameter(
                torch.randn(config.n_virtual_channels, n_params_pos_dim)
            )
        self.invalid_value = config.invalid_value
        self.heads.data /= pos_dim**0.5  # XXX Double check
        self.dropout = config.dropout
        self.dropout_around_channel = config.dropout_around_channel
        self.usage_penalty = config.usage_penalty
        self._penalty = torch.tensor(0.0)
        self.unmerge = config.unmerge

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def _get_weights(
        self,
        subject_ids: torch.Tensor,
        positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        B, C, _ = positions.shape
        if self.embed_ref:
            if positions.shape[2] == self.n_dims:
                ref_pad = torch.full(
                    (B, C, self.n_dims),
                    self.invalid_value,
                    dtype=positions.dtype,
                    device=positions.device,
                )
                positions = torch.cat([positions, ref_pad], dim=2)
            if positions.shape[2] != self.n_dims * 2:  # type: ignore
                got = positions.shape[2]
                raise ValueError(f"embed_ref needs {self.n_dims * 2} dims, got {got}")
            embedding = torch.cat(
                [
                    self.embedding(positions[..., : self.n_dims]),  # type: ignore
                    self.embedding(positions[..., self.n_dims :]),  # type: ignore
                ],
                dim=2,
            )
        else:
            if positions.shape[2] != self.n_dims:
                raise ValueError(
                    f"Expected {self.n_dims} spatial dimensions, got {positions.shape[2]}"
                )
            embedding = self.embedding(positions)

        score_offset = torch.zeros(B, C, device=device)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset = score_offset.masked_fill(invalid_mask, float("-inf"))

        if self.training and self.dropout:
            if self.unmerge:
                raise NotImplementedError(
                    "Figure out how to apply dropout if unmerge=True"
                )
            if self.dropout_around_channel:
                all_valid_positions = positions[~invalid_mask]
                ind = int(torch.randint(0, all_valid_positions.shape[0], (1,))[0])
                center_to_ban = all_valid_positions[ind, : self.n_dims].to(device)
            else:
                center_to_ban = torch.rand(self.n_dims, device=device)
            radius_to_ban = self.dropout
            banned = (positions[:, :, : self.n_dims] - center_to_ban).norm(
                dim=-1
            ) <= radius_to_ban
            score_offset = score_offset.masked_fill(banned, float("-inf"))

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            heads = self.heads.gather(
                0, subject_ids.view(-1, 1, 1).expand(-1, cout, pos_dim)
            )
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        if self.unmerge:
            scores = scores.transpose(1, 2)
        return torch.softmax(scores, dim=2).nan_to_num()  # Replace nans by 0

    def forward(
        self,
        meg: torch.Tensor,
        subject_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spatial attention on input.

        Parameters
        ----------
        positions :
            Normalized (x, y) coordinates for each channel in `meg`, of shape (B, C, 2).
            If shape is (B, C, 4), the additional two coordinates per channel indicate the
            position of the reference electrode. See `ns.extractors.ChannelPositions`.
        """
        weights = self._get_weights(subject_ids, positions, meg.device)
        out = weights @ meg
        if self.training and self.usage_penalty > 0.0:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = torch.tensor(self.usage_penalty * usage, device=meg.device)
        return out


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class UnitNorm(nn.Module):
    """Normalize last dimension of tensor to have unit Frobenius norm.

    Useful for parametrizing different normalization alternatives in `Mlp` below.

    NOTE: `hidden_dim` argument included for consistency with other normalization layers (e.g.
          BatchNorm).
    """

    def __init__(self, hidden_dim: int = 0) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p="fro", dim=-1, keepdim=True)


class Mean(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class NormDenormScaler(nn.Module):
    """Norm-denorm scaler inspired by [1]_.

    At inference time, this module applies z-score normalization of its input, followed by
    de-normalization based on the statistics of the data seen at instantiation.

    Parameters
    ----------
    x :
        Data on which to fit the denormalizer, of shape (n_examples, n_features).
    affine :
        If True, de-normalize with the statistics of `x`.

    References
    ----------
    .. [1] Ozcelik, Furkan, and Rufin VanRullen. "Natural scene reconstruction from fMRI signals
       using generative latent diffusion." Scientific Reports 13.1 (2023): 15666.
    """

    def __init__(self, x: torch.Tensor, affine: bool = True):
        super().__init__()

        if x.ndim != 2:
            raise ValueError(f"Tensor must be 2D (flattened), got ndim={x.ndim}")

        self.scaler = nn.BatchNorm1d(
            x.shape[1], affine=affine, track_running_stats=False, eps=1e-15
        ).eval()

        if affine:
            # Disable gradient as this is not currently intended to be finetuned
            self.scaler.weight.requires_grad = False
            self.scaler.bias.requires_grad = False

            self.scaler.weight.data = x.std(dim=0, correction=0)
            self.scaler.bias.data = x.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler(x)


class Mlp(BaseModelConfig):
    """Multilayer perceptron, e.g. for use as projection head.

    Notes
    -----
    Input size can be specified in the config or at build time.
    Output size can be specified at build time through the `output_size` parameter: it will be
    appended to `hidden_sizes` as the final layer.
    When `hidden_sizes` is empty or None and `output_size` is given, `build()` returns a
    single ``nn.Linear`` layer. When both are absent it returns ``nn.Identity``.
    """

    input_size: int | None = None
    hidden_sizes: list[int] | None = None

    norm_layer: tp.Literal["layer", "batch", "instance", "unit", None] = None
    activation_layer: tp.Literal["relu", "gelu", "elu", "prelu", None] = "relu"

    bias: bool = True
    dropout: float = 0.0

    @staticmethod
    def _get_norm_layer(kind: str | None) -> tp.Type[nn.Module] | None:
        return {
            "batch": nn.BatchNorm1d,
            "layer": nn.LayerNorm,
            "instance": nn.InstanceNorm1d,
            "unit": UnitNorm,
            None: None,
        }[kind]

    @staticmethod
    def _get_activation_layer(kind: str | None) -> tp.Type[nn.Module]:
        return {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            None: nn.Identity,
        }[kind]

    def build(
        self, input_size: int | None = None, output_size: int | None = None
    ) -> nn.Sequential | nn.Linear | nn.Identity:
        input_size = self.input_size if input_size is None else input_size
        if input_size is None:
            raise ValueError("input_size cannot be None.")
        if not self.hidden_sizes:
            if output_size is None:
                return nn.Identity()
            return nn.Linear(input_size, output_size)

        hidden_sizes = self.hidden_sizes.copy()
        if output_size is not None:
            hidden_sizes.append(output_size)

        return MLP(
            in_channels=input_size,
            hidden_channels=hidden_sizes,
            norm_layer=self._get_norm_layer(self.norm_layer),
            activation_layer=self._get_activation_layer(self.activation_layer),
            bias=self.bias,
            dropout=self.dropout,
        )


class TemporalDownsampling(BaseModelConfig):
    """Temporal downsampling via a 2-D convolution over the time axis.

    Parameters
    ----------
    kernel_size : int
        Kernel height (time-axis) of the downsampling convolution.
    stride : int
        Stride along the time axis.
    layer_norm : bool
        Apply ``LayerNorm`` after the convolution.
    layer_norm_affine : bool
        Use learnable affine parameters in the ``LayerNorm``.
    gelu : bool
        Apply GELU activation after normalization.
    """

    kernel_size: int = 45
    stride: int = 45
    layer_norm: bool = True
    layer_norm_affine: bool = True
    gelu: bool = True

    def build(self, dim: int) -> nn.Module:
        return TemporalDownsamplingModel(dim=dim, config=self)


class TemporalDownsamplingModel(nn.Module):
    """``nn.Module`` implementation of :class:`TemporalDownsampling`."""

    def __init__(
        self,
        dim: int,
        config: TemporalDownsampling | None = None,
    ) -> None:
        config = config if config is not None else TemporalDownsampling()
        super().__init__()

        self.agg = nn.Conv2d(
            1, 1, kernel_size=(config.kernel_size, 1), stride=(config.stride, 1)
        )
        self.layer_norm = (
            nn.LayerNorm(dim, eps=1e-8, elementwise_affine=config.layer_norm_affine)
            if config.layer_norm
            else None
        )
        self.gelu = nn.GELU() if config.gelu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape expected to be (B, 1, T, F)
        if x.ndim != 4:
            raise ValueError(f"Input must be of shape (B, 1, T, F), got {x.shape}")
        x = self.agg(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.gelu:
            x = self.gelu(x)
        return x
