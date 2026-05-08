# `emg/qwerty` — surface-EMG → QWERTY keystroke decoding (CTC)

Neuralbench task for the emg2qwerty benchmark (Sivakumar et al., NeurIPS
2024).  Inputs are 32-channel surface EMG at 2 kHz from two 16-electrode
wristbands; the model decodes a sequence of keystrokes from a 4-second
window using CTC.

See `docs/neuralbench/tasks/emg/qwerty.rst` for the full task page on
the docs site (description, dataset notes, citation).

## Quick smoke (5 epochs)

```bash
neuralbench --device emg --tasks qwerty --models emg2qwerty --download   # 1. fetch NM000104 (~239 GB) via eegdash
neuralbench --device emg --tasks qwerty --models emg2qwerty --debug      # 2. local sanity-check run
```

CER stays at 100 % at this scale — 5 epochs is well below the CTC
training horizon, so the model converges to all-blank without warmup.
This matches the upstream reference at the same scale.

## Recipes

Three `--dataset` overlays sit on top of `config.yaml`'s shared
optimizer + segmentation:

| Overlay | Output classes | Use when |
|---|---|---|
| `paper_personalized` | 99 | Train on N-1 sessions of one subject. |
| `paper_generic` | 99 | Cross-subject baseline (paper table 4). |
| `paper_personalized_compact` | 51 | Compact case + US-QWERTY shift-folded vocabulary. |

```bash
neuralbench --device emg --tasks qwerty --models emg2qwerty --dataset paper_personalized
```

Each overlay file has a `query:` field at the top — edit it to switch
subjects (or widen the cross-subject filter).

## Compact 51-class vocabulary

`paper_personalized_compact.yaml` switches `data.target.vocab_preset` to
`qwerty_compact`:

* uppercase letters → lowercase (`A` → `a`)
* US-QWERTY shifted digits → digits (`!` → `1`, …, `)` → `0`)
* US-QWERTY shifted punctuation → unshifted (`~` → `` ` ``, `_` → `-`,
  `+` → `=`, `{` → `[`, `}` → `]`, `|` → `\`, `:` → `;`, `"` → `'`,
  `<` → `,`, `>` → `.`, `?` → `/`)
* `Key.shift` modifier dropped

Roughly halves the output space; the trade-off is that the raw decoder
output loses shift-state information (recover via a downstream LM if
needed).

## Hardware notes

* `nn.CTCLoss` is **not implemented on MPS** (PyTorch ≤ 2.11) — the
  pipeline sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so CTC falls back to
  CPU while the rest stays on MPS.
* Paper numbers need GPU compute (A100/H100 single-GPU ≈ 6–8 h per
  generic baseline).  CPU / MPS are not feasible at full scale.

## Pretrained weights

Pre-remapped weights for the upstream `generic.ckpt` (108-subject
baseline) are published at
[`braindecode/emg2qwerty-generic`](https://huggingface.co/braindecode/emg2qwerty-generic)
under the upstream **CC BY-NC-SA 4.0** license:

```python
from braindecode.models import EMG2QwertyNet

model = EMG2QwertyNet.from_pretrained("braindecode/emg2qwerty-generic")
```

The two-key rename (`model.4.{weight,bias}` →
`final_layer.{weight,bias}`) and re-publishing flow live in
[`scripts/convert_emg2qwerty_checkpoint.py`](../../../scripts/convert_emg2qwerty_checkpoint.py)
— re-run it to produce additional remapped checkpoints.
