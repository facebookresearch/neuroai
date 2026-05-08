# `emg/qwerty` — surface-EMG → QWERTY keystroke decoding (CTC)

Neuralbench task for the emg2qwerty benchmark (Sivakumar et al., NeurIPS 2024).
Inputs are 32-channel surface EMG at 2 kHz from two 16-electrode wristbands;
the model decodes a sequence of keystrokes from a 4-second window using CTC.

## Files

| File | What it provides |
|---|---|
| `study.py` | `Emg2qwerty` — neuralset study source over the NM000104 BIDS tree. `Emg2qwertyRaw(Emg)` overrides BDF channel types (BIDS sidecar info MNE ignores) and rescales volts → microvolts to match the upstream HDF5 storage units (essential — the spectrogram log-floor caps gradients otherwise). |
| `extractors.py` | `KeystrokeSequence` — variable-length CTC target extractor. Pads each segment to `(max_target_length + 1,)` with the un-padded length in column 0. Optional `core_start_offset` / `core_duration` filters target events to the un-padded core of a padded EMG window. |
| `charset.py` | Vocabulary presets. ``paper`` (default): 98 keys (letters + digits + punctuation + 4 modifiers), ``num_classes = 99``. ``qwerty_compact``: 50 keys (lowercase letters + digits + 11 unshifted punctuation + 3 modifiers; case + shift folded), ``num_classes = 51``. Switch via the ``vocab_preset`` field on ``KeystrokeSequence``. |
| `metrics.py` | `CharacterErrorRates` — Levenshtein-based CER torchmetric; insertions / deletions / substitutions are kept on the metric for post-hoc IER/DER/SER inspection. |
| `callbacks.py` | `SpecAugmentCallback` (forward hook on `model.spectrogram`), `BandRotationCallback` (electrode roll + temporal jitter), and matching `BaseCallbackConfig` wrappers (`SpecAugment`, `BandRotation`) for YAML wiring. |
| `config.yaml`, `datasets/*.yaml` | Task and recipe configs for the neuralbench CLI. |

## Quick smoke (5 epochs)

```bash
neuralbench --device emg --tasks qwerty --models emg2qwerty --download   # 1. fetch NM000104 (~239 GB)
neuralbench --device emg --tasks qwerty --models emg2qwerty --debug      # 2. local sanity-check run
```

The dataset is auto-fetched from NEMAR (`s3://nemar/nm000104`) via
`neuralfetch.download.Eegdash` (1136 files, ~239 GB). It lands under
`<DATA_DIR>/Emg2qwerty/download/sub-*/ses-*/emg/`. If you already have
the BIDS tree placed directly under `<DATA_DIR>/Emg2qwerty/`, the loader
picks that up too.

Pipeline completes; CER stays at 100% — 5 epochs is well below the CTC
training horizon, so the model converges to all-blank without warmup.

## Paper recipes

The optimizer (Adam + OneCycleLR, lr 1e-8 → 1e-3 → 1e-6, 40% warmup)
lives in `config.yaml`; the augmentation callbacks live in
`models/emg2qwerty.yaml`.  The four `--dataset` overlays carry the
deltas needed for each regime:

| Overlay | Vocabulary | Init | Use when |
|---|---|---|---|
| `paper_personalized` | 99 (paper) | random | Reproducing the paper's per-subject CER from scratch. |
| `paper_generic` | 99 (paper) | random | Cross-subject baseline (paper table 4). |
| `paper_personalized_finetune` | 99 (paper) | upstream `--checkpoint` | Fine-tuning from upstream emg2qwerty pretrained weights. |
| `paper_personalized_compact` | 51 (compact) | random | From-scratch with the case-folded + shift-folded vocabulary. |

```bash
# 1. Personalized from scratch (paper-faithful 99-class vocabulary).
neuralbench --device emg --tasks qwerty --models emg2qwerty \
    --dataset paper_personalized

# 2. Generic cross-subject baseline.
neuralbench --device emg --tasks qwerty --models emg2qwerty \
    --dataset paper_generic

# 3. Fine-tune from the upstream generic checkpoint (best CER).
neuralbench --device emg --tasks qwerty --models emg2qwerty \
    --dataset paper_personalized_finetune \
    --checkpoint /path/to/upstream_generic.ckpt

# 4. Personalized from scratch with the compact 51-class vocabulary.
neuralbench --device emg --tasks qwerty --models emg2qwerty \
    --dataset paper_personalized_compact
```

Each overlay file has a `>>> EDIT THE SUBJECT BELOW <<<` marker on the
query line for switching subjects.

Paper headline numbers (table 4): generic baseline ≈ 30% CER (150 epochs);
personalized (+ 50-epoch fine-tune on one subject) ≈ 10% CER.

## Fine-tuning from upstream pretrained weights

The upstream emg2qwerty release ships generic-pretrained
`TDSConvCTCModule` checkpoints. They load directly into braindecode's
`EMG2QwertyNet` via `--checkpoint`: 49 of 51 keys match verbatim, and
`EMG2QwertyNet.mapping` declares the 2-key rename
(`model.4.{weight,bias}` → `final_layer.{weight,bias}`) that
`neuralbench.utils.load_checkpoint` applies on the way in.

The `paper_personalized_finetune.yaml` overlay lowers `max_lr` by 100×
(1e-3 → 1e-5), shortens the warmup (`pct_start` 0.4 → 0.1), and defers
augmentation start (`SpecAugment` from epoch 5, `BandRotation` from
epoch 3). The from-scratch recipe overshoots the pretrained minimum
within a few epochs — use this overlay whenever `--checkpoint` is set.

Empirical sanity check on an A40 (sub-01438774, 10 epochs): zero-shot
val/CER from the upstream generic checkpoint is ≈ 16 %; the from-scratch
recipe applied to the same init drives val/CER to 75 % within 5 epochs.

## Compact 51-class vocabulary

The `paper_personalized_compact.yaml` overlay sets
`data.target.vocab_preset: qwerty_compact` and reduces the model's
output dim from 99 to 51 by folding the target labels:

* uppercase letters → lowercase (`A` → `a`)
* US-QWERTY shifted digits → digits (`!` → `1`, `@` → `2`, …, `)` → `0`)
* US-QWERTY shifted punctuation → unshifted (`~` → `` ` ``, `_` → `-`,
  `+` → `=`, `{` → `[`, `}` → `]`, `|` → `\`, `:` → `;`, `"` → `'`,
  `<` → `,`, `>` → `.`, `?` → `/`)
* `Key.shift` modifier dropped (now redundant)

Roughly halves the output space, which gives more samples per class and
trains faster.  The trade-off: the raw decoder output loses
shift-state information, so capitalization / shifted symbols need to be
recovered downstream by a language model or post-processor if the
target use case requires them.

Not compatible with `--checkpoint` against upstream's generic
checkpoint (its 99-class classifier head won't load into a 51-class
model — sizes mismatch).  Train from scratch.

## Test-time windowing caveat

The upstream paper trains on fixed 4-s windows (0.9 s left + 0.1 s
right padding) but feeds **whole sessions** at test time
([`emg2qwerty/lightning.py:87-100`](https://github.com/facebookresearch/emg2qwerty/blob/main/emg2qwerty/lightning.py))
"for more realism".  Our pipeline's segmenter applies the same 4-s
window at test time as well — slightly pessimistic for CER reporting.
Closing this gap requires a per-split duration override on neuralset's
segmenter; tracked as a follow-up.

## Hardware notes

* `nn.CTCLoss` is **not implemented on MPS** (PyTorch ≤ 2.11). The pipeline
  sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so the rest of the model runs on
  MPS while CTC falls back to CPU.
* Reaching the paper numbers needs GPU compute — A100/H100 single-GPU is
  ~6–8 h per generic-baseline run; CPU-only is not feasible (~weeks per
  run); MPS is not feasible (CTC fallback dominates per-step cost).
* Smoke runs on Apple Silicon converge to all-blank within 1 epoch
  regardless of recipe, matching the upstream `emg2qwerty.train`
  reference run on the same data scale.
