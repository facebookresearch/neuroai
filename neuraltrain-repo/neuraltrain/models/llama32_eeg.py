# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Task-conditioned Llama 3.2 backbone for NeuralBench EEG tasks."""

from __future__ import annotations

import json
import typing as tp
from itertools import combinations

import torch
from torch import nn

from .base import BaseBrainModelConfig


class Llama32EEG(BaseBrainModelConfig):
    """Use frozen Llama hidden states with a learned EEG prefix and output head.

    The benchmark supplies the output width from its target extractor. Its
    existing loss and metrics then define classification, multilabel,
    regression, or retrieval behavior without model-specific task mappings.
    """

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    revision: str = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    prompt_template: str = "Analyze this EEG recording for the {task} task."
    n_eeg_tokens: int = 16
    torch_dtype: tp.Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def build(
        self,
        n_spatial_locations: int,
        n_outputs: int | None,
        task_name: str | None = None,
    ) -> nn.Module:
        if n_outputs is None or n_outputs < 1:
            raise ValueError("Llama32EEG requires a positive output width")
        if self.n_eeg_tokens < 1:
            raise ValueError("n_eeg_tokens must be positive")
        task = (task_name or "EEG decoding").replace("_", " ")
        return Llama32EEGModel(
            n_channels=n_spatial_locations,
            n_outputs=n_outputs,
            model_name=self.model_name,
            revision=self.revision,
            prompt=self.prompt_template.format(task=task),
            n_eeg_tokens=self.n_eeg_tokens,
            torch_dtype=self.torch_dtype,
        )


class Llama32EEGModel(nn.Module):
    """Predict benchmark targets from EEG-conditioned Llama representations."""

    # This model has no lazy layers. The factory's CPU dummy forwards are
    # unnecessary and prohibitively expensive for a 3B-parameter backbone.
    skip_dummy_forward = True
    skip_torchinfo_summary = True
    omit_frozen_backbone_from_checkpoint = True

    def __init__(
        self,
        *,
        n_channels: int,
        n_outputs: int,
        model_name: str,
        revision: str | None = None,
        prompt: str,
        n_eeg_tokens: int,
        torch_dtype: str,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("Llama chat template produced an empty prompt")
        self.register_buffer("prompt_ids", torch.tensor(prompt_ids, dtype=torch.long))
        self.llama = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=getattr(torch, torch_dtype),
            low_cpu_mem_usage=True,
        )
        self.llama.requires_grad_(False)
        self.llama.config.use_cache = False
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        hidden_size = self.llama.config.hidden_size
        self.eeg_pool = nn.AdaptiveAvgPool1d(n_eeg_tokens)
        self.eeg_projector = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.output_head = nn.Linear(hidden_size, n_outputs)

    def train(self, mode: bool = True) -> Llama32EEGModel:
        super().train(mode)
        self.llama.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected EEG (batch, channels, time), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.eeg_projector[0].normalized_shape[0]:
            raise ValueError("EEG channel count differs from model build context")
        eeg = self.eeg_pool(x.float()).transpose(1, 2)
        eeg_embeds = self.eeg_projector(eeg)
        text_embeds = self.llama.get_input_embeddings()(
            self.prompt_ids.unsqueeze(0).expand(x.shape[0], -1)
        )
        inputs_embeds = torch.cat([eeg_embeds.to(text_embeds.dtype), text_embeds], dim=1)
        hidden = self.llama(
            inputs_embeds=inputs_embeds, use_cache=False
        ).last_hidden_state
        return self.output_head(hidden[:, -1].float())


def _parse_multilabel_json(text: str, labels: list[str]) -> tuple[list[float], bool]:
    """Decode an exact event list; invalid generations receive no labels."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return [0.0] * len(labels), False
    if not isinstance(value, list) or any(
        not isinstance(label, str) or label not in labels for label in value
    ):
        return [0.0] * len(labels), False
    if len(value) != len(set(value)):
        return [0.0] * len(labels), False
    selected = set(value)
    return [float(label in selected) for label in labels], True


class Llama32EEGGenerate(BaseBrainModelConfig):
    """Generate a JSON event list for a multilabel NeuralBench EEG task."""

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    revision: str = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    n_eeg_tokens: int = 16
    max_new_tokens: int = 64
    constrain_event_list: bool = True
    torch_dtype: tp.Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def build(
        self,
        n_spatial_locations: int,
        n_outputs: int | None,
        target_labels: list[str] | None = None,
        task_name: str | None = None,
    ) -> nn.Module:
        if not target_labels or n_outputs != len(target_labels):
            raise ValueError("Generative EEG evaluation needs ordered target labels")
        if self.n_eeg_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("EEG prefix and generation lengths must be positive")
        return Llama32EEGGenerateModel(
            n_channels=n_spatial_locations,
            labels=target_labels,
            task_name=task_name or "EEG decoding",
            model_name=self.model_name,
            revision=self.revision,
            n_eeg_tokens=self.n_eeg_tokens,
            max_new_tokens=self.max_new_tokens,
            constrain_event_list=self.constrain_event_list,
            torch_dtype=self.torch_dtype,
        )


class Llama32EEGGenerateModel(nn.Module):
    """Train an EEG prefix with teacher forcing, evaluate generated event lists."""

    skip_dummy_forward = True
    skip_torchinfo_summary = True
    omit_frozen_backbone_from_checkpoint = True

    def __init__(
        self,
        *,
        n_channels: int,
        labels: list[str],
        task_name: str,
        model_name: str,
        revision: str | None,
        n_eeg_tokens: int,
        max_new_tokens: int,
        constrain_event_list: bool,
        torch_dtype: str,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.labels = labels
        self.max_new_tokens = max_new_tokens
        self.constrain_event_list = constrain_event_list
        self.last_generated_texts: list[str] = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        allowed = json.dumps(labels, ensure_ascii=False)
        prompt = (
            f"Analyze this EEG recording for the {task_name.replace('_', ' ')} task. "
            f"Return only a JSON array containing every present event from {allowed}. "
            "Use [] if none are present. Do not add prose."
        )
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("Llama chat template produced an empty prompt")
        self.register_buffer("prompt_ids", torch.tensor(prompt_ids, dtype=torch.long))
        self._event_token_trie: dict[int, tp.Any] = {}
        if constrain_event_list:
            for count in range(len(labels) + 1):
                for chosen in combinations(labels, count):
                    response = json.dumps(
                        chosen, ensure_ascii=False, separators=(",", ":")
                    )
                    ids = self.tokenizer.encode(response, add_special_tokens=False)
                    ids.append(self.tokenizer.eos_token_id)
                    if len(ids) > max_new_tokens:
                        raise ValueError("max_new_tokens cannot fit every event list")
                    node = self._event_token_trie
                    for token_id in ids:
                        node = node.setdefault(token_id, {})
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=getattr(torch, torch_dtype),
            low_cpu_mem_usage=True,
        )
        self.llama.requires_grad_(False)
        self.llama.config.use_cache = False
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        hidden_size = self.llama.config.hidden_size
        self.eeg_pool = nn.AdaptiveAvgPool1d(n_eeg_tokens)
        self.eeg_projector = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def train(self, mode: bool = True) -> Llama32EEGGenerateModel:
        super().train(mode)
        self.llama.eval()
        return self

    def _prefix(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.eeg_projector[0].normalized_shape[0]:
            raise ValueError("Expected EEG with the channel count in the build context")
        eeg = self.eeg_pool(x.float()).transpose(1, 2)
        eeg_embeds = self.eeg_projector(eeg)
        prompt_embeds = self.llama.get_input_embeddings()(
            self.prompt_ids.unsqueeze(0).expand(x.shape[0], -1)
        )
        return torch.cat([eeg_embeds.to(prompt_embeds.dtype), prompt_embeds], dim=1)

    def _responses(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if targets.ndim != 2 or targets.shape[1] != len(self.labels):
            raise ValueError("Generative multilabel targets must be (batch, labels)")
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer needs an EOS token")
        rows = []
        for target in targets.detach().cpu().tolist():
            selected = [label for label, value in zip(self.labels, target) if value > 0.5]
            response = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
            rows.append(self.tokenizer.encode(response, add_special_tokens=False) + [eos])
        tokens = torch.full(
            (len(rows), max(map(len, rows))), eos, dtype=torch.long, device=targets.device
        )
        mask = torch.zeros_like(tokens)
        for index, row in enumerate(rows):
            tokens[index, : len(row)] = torch.tensor(row, device=targets.device)
            mask[index, : len(row)] = 1
        return tokens, mask

    def _language_loss(self, prefix: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tokens, response_mask = self._responses(targets)
        response_embeds = self.llama.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat([prefix, response_embeds], dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        attention_mask = torch.cat([prefix_mask, response_mask], dim=1)
        labels = torch.cat(
            [
                torch.full_like(prefix_mask, -100),
                tokens.masked_fill(~response_mask.bool(), -100),
            ],
            dim=1,
        )
        return self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss

    @torch.no_grad()
    def _generate(self, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def allowed_tokens(_: int, token_ids: torch.Tensor) -> list[int]:
            # With inputs_embeds, the constraint sees only newly generated IDs.
            node = self._event_token_trie
            for token_id in token_ids.tolist():
                if not node:
                    return [self.tokenizer.eos_token_id]
                node = node[token_id]
            if not node:
                # HF still queries finished rows while other batch rows decode.
                return [self.tokenizer.eos_token_id]
            return list(node)

        generated = self.llama.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(
                prefix.shape[:2], dtype=torch.long, device=prefix.device
            ),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            prefix_allowed_tokens_fn=allowed_tokens
            if self.constrain_event_list
            else None,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        self.last_generated_texts = decoded
        parsed = [_parse_multilabel_json(value, self.labels) for value in decoded]
        scores = torch.tensor(
            [[10.0 if active else -10.0 for active in values] for values, _ in parsed],
            dtype=torch.float32,
            device=prefix.device,
        )
        invalid_fraction = torch.tensor(
            sum(not valid for _, valid in parsed) / len(parsed), device=prefix.device
        )
        return scores, invalid_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores, _ = self._generate(self._prefix(x))
        return scores

    def generative_step(
        self, x: torch.Tensor, targets: torch.Tensor, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prefix = self._prefix(x)
        language_loss = self._language_loss(prefix, targets)
        if stage == "train":
            return torch.zeros_like(targets), language_loss, None
        scores, invalid_fraction = self._generate(prefix.detach())
        return scores, language_loss, invalid_fraction
