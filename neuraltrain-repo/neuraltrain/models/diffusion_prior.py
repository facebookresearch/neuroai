# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Diffusion prior used in MindEye (https://github.com/MedARC-AI/fMRI-reconstruction-NSD/tree/main).
"""

import logging
import random
import typing as tp

import torch
from torch import nn
from tqdm import tqdm

from .base import BaseModelConfig

logger = logging.getLogger(__name__)

try:
    import dalle2_pytorch.dalle2_pytorch as dalle2_modules
    from dalle2_pytorch import DiffusionPrior as DalleDiffusionPrior

    class PriorNetwork(nn.Module):  # type : ignore
        def __init__(
            self,
            dim: int,
            num_timesteps: int | None = None,  # type: ignore
            num_time_embeds: int = 1,
            num_text_tokens: int = 257,
            num_image_tokens: int = 257,
            causal: bool = True,
            learned_query_mode: str = "none",
            **kwargs,
        ):
            super().__init__()
            self.dim = dim
            self.num_time_embeds = num_time_embeds
            self.continuous_embedded_time = not dalle2_modules.exists(num_timesteps)
            self.learned_query_mode = learned_query_mode

            self.to_time_embeds = nn.Sequential(
                (
                    nn.Embedding(num_timesteps, dim * num_time_embeds)  # type: ignore
                    if dalle2_modules.exists(num_timesteps)
                    else nn.Sequential(
                        dalle2_modules.SinusoidalPosEmb(dim),
                        dalle2_modules.MLP(dim, dim * num_time_embeds),
                    )
                ),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
                dalle2_modules.Rearrange("b (n d) -> b n d", n=num_time_embeds),
            )

            if self.learned_query_mode == "token":
                self.learned_query = nn.Parameter(torch.randn(num_image_tokens, dim))
            if self.learned_query_mode == "pos_emb":
                scale = dim**-0.5
                self.learned_query = nn.Parameter(
                    torch.randn(num_image_tokens, dim) * scale
                )
            if self.learned_query_mode == "all_pos_emb":
                scale = dim**-0.5
                self.learned_query = nn.Parameter(
                    torch.randn(num_image_tokens + num_text_tokens + 1, dim) * scale
                )
            self.causal_transformer = FlaggedCausalTransformer(
                dim=dim, causal=causal, **kwargs
            )

            self.null_text_embeds = nn.Parameter(torch.randn(num_text_tokens, dim))
            self.null_image_embed = nn.Parameter(torch.randn(num_image_tokens, dim))

            self.num_image_tokens = num_image_tokens
            self.num_text_tokens = num_text_tokens
            self.self_cond = False

        def forward_with_cond_scale(self, *args, cond_scale: float = 1.0, **kwargs):
            logits = self.forward(*args, **kwargs)

            if cond_scale == 1.0:
                return logits

            null_logits = self.forward(
                *args,
                text_cond_drop_prob=1.0,
                image_cond_drop_prob=1,
                **kwargs,  # type: ignore
            )
            return null_logits + (logits - null_logits) * cond_scale

        def forward(
            self,
            image_embed: torch.Tensor,  # image_embed is the target we aim to denoise with diffusion network
            diffusion_timesteps: torch.Tensor,
            text_embed: torch.Tensor,  # text_embed are conditioning inputs of diffusion model
            text_cond_drop_prob: float = 0.0,
            image_cond_drop_prob: float = 0.0,
            **kwargs,
        ):
            # text_embed = text_embed
            # brain_cond_drop_prob = text_cond_drop_prob

            image_embed = image_embed.view(len(image_embed), -1, self.dim)
            text_embed = text_embed.view(len(text_embed), -1, self.dim)

            batch, _, dim, device, dtype = (
                *image_embed.shape,
                image_embed.device,
                image_embed.dtype,
            )

            # classifier free guidance masks
            text_keep_mask = dalle2_modules.prob_mask_like(
                (batch,), 1 - text_cond_drop_prob, device=device
            )
            text_keep_mask = dalle2_modules.rearrange(text_keep_mask, "b -> b 1 1")

            image_keep_mask = dalle2_modules.prob_mask_like(
                (batch,), 1 - image_cond_drop_prob, device=device
            )
            image_keep_mask = dalle2_modules.rearrange(image_keep_mask, "b -> b 1 1")

            # mask out text embeddings with null text embeddings
            null_text_embeds = self.null_text_embeds.to(text_embed.dtype)
            text_embed = torch.where(text_keep_mask, text_embed, null_text_embeds[None])

            # mask out image embeddings with null image embeddings
            null_image_embed = self.null_image_embed.to(image_embed.dtype)
            image_embed = torch.where(
                image_keep_mask, image_embed, null_image_embed[None]
            )

            if self.continuous_embedded_time:
                diffusion_timesteps = diffusion_timesteps.type(dtype)
            time_embed = self.to_time_embeds(diffusion_timesteps)

            if self.learned_query_mode == "token":
                learned_queries = dalle2_modules.repeat(
                    self.learned_query, "n d -> b n d", b=batch
                )
            elif self.learned_query_mode == "pos_emb":
                pos_embs = dalle2_modules.repeat(
                    self.learned_query, "n d -> b n d", b=batch
                )
                image_embed = image_embed + pos_embs
                learned_queries = torch.empty((batch, 0, dim), device=text_embed.device)
            elif self.learned_query_mode == "all_pos_emb":
                pos_embs = dalle2_modules.repeat(
                    self.learned_query, "n d -> b n d", b=batch
                )
                learned_queries = torch.empty((batch, 0, dim), device=text_embed.device)
            else:
                learned_queries = torch.empty((batch, 0, dim), device=text_embed.device)

            tokens = torch.cat(
                (text_embed, time_embed, image_embed, learned_queries), dim=-2
            )

            if self.learned_query_mode == "all_pos_emb":
                tokens = tokens + pos_embs

            # attend
            tokens = self.causal_transformer(tokens)

            # get learned query, which should predict the image embedding (per DDPM timestep)
            pred_image_embed = tokens[..., -self.num_image_tokens :, :]

            return pred_image_embed

    class NewDiffusionPrior(DalleDiffusionPrior):  # type : ignore
        @torch.no_grad()
        def p_sample(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            text_cond: tp.Dict[str, torch.Tensor] | None = None,
            self_cond: torch.Tensor | None = None,
            clip_denoised: bool = True,
            cond_scale: float = 1.0,
            generator: torch.Generator | None = None,
        ):
            b, *_, device = *x.shape, x.device
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(
                x=x,
                t=t,
                text_cond=text_cond,
                self_cond=self_cond,
                clip_denoised=clip_denoised,
                cond_scale=cond_scale,
            )
            if generator is None:
                noise = torch.randn_like(x)
            else:
                noise = torch.randn(
                    x.size(), device=device, dtype=x.dtype, generator=generator
                )
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
            pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            return pred, x_start

        @torch.no_grad()
        def p_sample_loop_ddpm(
            self,
            shape: tuple[int],
            text_cond: tp.Dict[str, torch.Tensor] | None = None,
            cond_scale: float = 1.0,
            generator: torch.Generator | None = None,
        ):
            batch, device = shape[0], self.device

            if generator is None:
                image_embed = torch.randn(shape, device=device)
            else:
                image_embed = torch.randn(shape, device=device, generator=generator)
            x_start = None  # for self-conditioning

            if self.init_image_embed_l2norm:
                image_embed = dalle2_modules.l2norm(image_embed) * self.image_embed_scale

            for i in tqdm(
                reversed(range(0, self.noise_scheduler.num_timesteps)),
                desc="sampling loop time step",
                total=self.noise_scheduler.num_timesteps,
                disable=True,
            ):
                times = torch.full((batch,), i, device=device, dtype=torch.long)

                self_cond = x_start if self.net.self_cond else None
                image_embed, x_start = self.p_sample(
                    image_embed,
                    times,
                    text_cond=text_cond,
                    self_cond=self_cond,
                    cond_scale=cond_scale,
                    generator=generator,
                )

            if self.sampling_final_clamp_l2norm and self.predict_x_start:
                image_embed = self.l2norm_clamp_embed(image_embed)

            return image_embed

        def p_losses(
            self,
            image_embed: torch.Tensor,
            times: torch.Tensor,
            text_cond: tp.Dict[str, torch.Tensor],
            noise: torch.Tensor | None = None,
        ):
            noise = dalle2_modules.default(noise, lambda: torch.randn_like(image_embed))

            image_embed_noisy = self.noise_scheduler.q_sample(
                x_start=image_embed, t=times, noise=noise
            )

            self_cond = None
            if self.net.self_cond and random.random() < 0.5:
                with torch.no_grad():
                    self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

            pred = self.net(
                image_embed_noisy,
                times,
                self_cond=self_cond,
                text_cond_drop_prob=self.text_cond_drop_prob,
                image_cond_drop_prob=self.image_cond_drop_prob,
                **text_cond,
            )

            if self.predict_x_start and self.training_clamp_l2norm:
                pred = self.l2norm_clamp_embed(pred)

            if self.predict_v:
                target = self.noise_scheduler.calculate_v(image_embed, times, noise)
            elif self.predict_x_start:
                target = image_embed
            else:
                target = noise
            return {"pred": pred, "target": target}

        def forward(
            self,
            text_embed: torch.Tensor,
            image_embed: torch.Tensor | None = None,
            text_encodings: torch.Tensor | None = None,
            *args,
            **kwargs,
        ):
            if self.training:
                if image_embed is None:
                    raise ValueError(
                        "image_embed should be passed to diffusion prior during training"
                    )
                text_cond = dict(text_embed=text_embed)

                if self.condition_on_text_encodings:
                    if not dalle2_modules.exists(text_encodings):
                        raise ValueError(
                            "text encodings must be present"
                            " for diffusion prior if specified"
                        )
                    text_cond = {**text_cond, "text_encodings": text_encodings}  # type: ignore

                # timestep conditioning from ddpm
                batch = image_embed.shape[0]
                times = self.noise_scheduler.sample_random_times(batch)

                # calculate forward loss
                out = self.p_losses(
                    image_embed * self.image_embed_scale,
                    times,
                    text_cond=text_cond,  # type: ignore
                    *args,
                    **kwargs,
                )
                return out
            else:
                if image_embed is not None:
                    raise ValueError(
                        "image_embed should not be passed to diffusion prior during evaluation/sampling mode, as generating conditioned on text_embed only, image embed is target during training"
                    )
                out_shape = (
                    text_embed.shape[0],
                    self.net.num_image_tokens,
                    self.image_embed_dim,
                )
                return self.p_sample_loop(
                    out_shape, text_cond=dict(text_embed=text_embed)
                )

    DiffusionPriorModel = NewDiffusionPrior  # type: ignore

    class FlaggedCausalTransformer(nn.Module):
        def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            norm_in=False,
            norm_out=True,
            attn_dropout=0.0,
            ff_dropout=0.0,
            final_proj=True,
            normformer=False,
            rotary_emb=True,
            causal=True,
        ):
            super().__init__()
            self.init_norm = dalle2_modules.LayerNorm(dim) if norm_in else nn.Identity()

            self.rel_pos_bias = dalle2_modules.RelPosBias(heads=heads)

            rotary_emb = (
                dalle2_modules.RotaryEmbedding(dim=min(32, dim_head))
                if rotary_emb
                else None
            )

            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            dalle2_modules.Attention(
                                dim=dim,
                                causal=causal,
                                dim_head=dim_head,
                                heads=heads,
                                dropout=attn_dropout,
                                rotary_emb=rotary_emb,
                            ),
                            dalle2_modules.FeedForward(
                                dim=dim,
                                mult=ff_mult,
                                dropout=ff_dropout,
                                post_activation_norm=normformer,
                            ),
                        ]
                    )
                )

            self.norm = (
                dalle2_modules.LayerNorm(dim, stable=True) if norm_out else nn.Identity()
            )
            self.project_out = (
                nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()
            )

        def forward(self, x: torch.Tensor):
            n, device = x.shape[1], x.device

            x = self.init_norm(x)

            attn_bias = self.rel_pos_bias(n, n + 1, device=device)

            for attn, ff in self.layers:
                x = attn(x, attn_bias=attn_bias) + x
                x = ff(x) + x

            out = self.norm(x)
            return self.project_out(out)

except ImportError:

    class DummyDiffusionPrior(torch.nn.Module):  # type : ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "Please install dalle2-pytorch to use DiffusionPrior: pip install dalle2-pytorch"
            )

    DiffusionPriorModel = DummyDiffusionPrior  # type: ignore

    class PriorNetwork(torch.nn.Module):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "Please install dalle2-pytorch to use DiffusionPrior: pip install dalle2-pytorch"
            )


class DiffusionPrior(BaseModelConfig):
    """Diffusion prior module adapted from MindEye [1]_.

    Although the parameters *text_embed* and *image_embed* appear to refer
    specifically to text and image data, they can represent any embedding:
    *text_embed* is the input (x) to the diffusion prior, and *image_embed*
    is the target (y) that the prior aims to denoise.

    Parameters
    ----------
    depth : int
        Number of Transformer layers in the prior network.
    dim_head : int
        Dimension per attention head.
    prior_learned_query_mode : {"token", "pos_emb", "all_pos_emb"}
        How to handle learned queries for image tokens.
    timesteps : int
        Number of diffusion denoising steps.
    cond_drop_prob : float
        Dropout probability applied to the conditioning input for
        classifier-free guidance.
    predict : {"x_start", "v"}
        Prediction target: ``"x_start"`` predicts the clean embedding
        directly; ``"v"`` uses the velocity parameterisation from Imagen.

    References
    ----------
    .. [1] https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/models.py
    """

    depth: int = 6
    dim_head: int = 64
    prior_learned_query_mode: tp.Literal["token", "pos_emb", "all_pos_emb"] = "pos_emb"
    timesteps: int = 100
    cond_drop_prob: float = 0.2

    # prediction type
    predict: tp.Literal["x_start", "v"] = "x_start"

    def build(
        self,
        dim: int,
        num_out_tokens: int,
        num_in_tokens: int,
    ) -> DiffusionPriorModel:
        if dim % self.dim_head != 0:
            raise ValueError(f"dim {dim} must be divisible by dim_head {self.dim_head}")
        heads = dim // self.dim_head

        prior_network = PriorNetwork(
            dim=dim,
            depth=self.depth,
            dim_head=self.dim_head,
            heads=heads,
            causal=False,
            num_image_tokens=num_out_tokens,
            num_text_tokens=num_in_tokens,
            learned_query_mode=self.prior_learned_query_mode,
        )
        logger.info("prior_network loaded")

        diffusion_prior = DiffusionPriorModel(
            net=prior_network,
            image_embed_dim=dim,
            condition_on_text_encodings=False,
            timesteps=self.timesteps,
            cond_drop_prob=self.cond_drop_prob,
            image_embed_scale=None,
            predict_x_start=True if self.predict == "x_start" else False,
            predict_v=True if self.predict == "v" else False,
        )
        return diffusion_prior
