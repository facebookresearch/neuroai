"""Small model test without downloading a gated Llama checkpoint."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from neuraltrain.models.llama32_eeg import Llama32EEG, Llama32EEGModel


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        assert kwargs["tokenize"] is False
        return "<prompt>"

    def encode(self, prompt, **kwargs):
        assert prompt == "<prompt>"
        assert kwargs["add_special_tokens"] is False
        return [1, 2, 3]


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 12, "use_cache": True})()
        self.embedding = nn.Embedding(8, 12)
        self.decoder = nn.GRU(12, 12, batch_first=True)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, *, inputs_embeds, use_cache):
        assert not use_cache
        states, _ = self.decoder(inputs_embeds)
        return type("Output", (), {"last_hidden_state": states})()


@pytest.mark.parametrize("n_outputs", [1, 2, 6, 29, 1536, 4096])
def test_llama32_eeg_output_width_and_gradients(monkeypatch, n_outputs):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: _LanguageModel()
    )
    model = Llama32EEGModel(
        n_channels=3,
        n_outputs=n_outputs,
        model_name="fake-llama",
        prompt="Analyze EEG",
        n_eeg_tokens=4,
        torch_dtype="float32",
    )
    model.train()
    logits = model(torch.randn(2, 3, 40))
    assert logits.shape == (2, n_outputs)
    assert torch.isfinite(logits).all()
    assert not model.llama.training
    logits.sum().backward()
    assert any(p.grad is not None for p in model.eeg_projector.parameters())
    assert model.output_head.weight.grad is not None
    assert all(p.grad is None for p in model.llama.parameters())


def test_llama32_uses_task_name_in_prompt(monkeypatch):
    import transformers

    prompts = []
    revisions = []

    class Tokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return super().apply_chat_template(messages, **kwargs)

    def load_tokenizer(*args, **kwargs):
        revisions.append(kwargs["revision"])
        return Tokenizer()

    def load_model(*args, **kwargs):
        revisions.append(kwargs["revision"])
        return _LanguageModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    model = Llama32EEG().build(
        n_spatial_locations=3, n_outputs=6, task_name="clinical_event"
    )
    assert "clinical event" in prompts[0]
    assert revisions == [Llama32EEG().revision] * 2
    assert model(torch.randn(2, 3, 20)).shape == (2, 6)


def test_llama32_with_tiny_huggingface_llama(monkeypatch):
    import transformers
    from transformers.models.llama.modeling_llama import LlamaModel

    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: LlamaModel(config)
    )
    model = Llama32EEGModel(
        n_channels=3,
        n_outputs=5,
        model_name="tiny-llama",
        prompt="Analyze EEG",
        n_eeg_tokens=4,
        torch_dtype="float32",
    )
    loss = model(torch.randn(2, 3, 40)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.eeg_projector[1].weight.grad is not None
    assert all(p.grad is None for p in model.llama.parameters())


@pytest.mark.parametrize(
    "task_kind", ["classification", "multilabel", "regression", "retrieval"]
)
def test_llama32_trains_with_neuralbench_eeg_losses(monkeypatch, task_kind):
    import transformers

    from neuraltrain.losses.losses import ClipLoss

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: _LanguageModel()
    )
    widths = {"classification": 3, "multilabel": 6, "regression": 1, "retrieval": 16}
    width = widths[task_kind]
    model = Llama32EEGModel(
        n_channels=3,
        n_outputs=width,
        model_name="fake-llama",
        prompt="Analyze EEG",
        n_eeg_tokens=4,
        torch_dtype="float32",
    )
    predictions = model(torch.randn(2, 3, 40))
    if task_kind == "classification":
        loss = nn.CrossEntropyLoss()(predictions, torch.eye(width)[:2])
    elif task_kind == "multilabel":
        loss = nn.BCEWithLogitsLoss()(
            predictions, torch.randint(0, 2, predictions.shape).float()
        )
    elif task_kind == "regression":
        loss = nn.MSELoss()(predictions, torch.randn_like(predictions))
    else:
        loss = ClipLoss(norm_kind="y", temperature=False, symmetric=False)(
            predictions, torch.randn_like(predictions)
        )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output_head.weight.grad is not None
    assert all(p.grad is None for p in model.llama.parameters())
