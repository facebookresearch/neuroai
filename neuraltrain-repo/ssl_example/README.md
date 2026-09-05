# `neuraltrain` self-supervised example

This directory contains an example project showing how to use `neuraltrain` and
`pytorch-lightning` to pretrain a masked autoencoder (MAE) on unlabelled EEG
data, and how to hand the resulting encoder to `neuralbench` for evaluation.

## Description

This example loads the EEG channels of the
[MNE sample dataset](https://mne.tools/stable/documentation/datasets.html#sample)
and pretrains a small [MAE](https://arxiv.org/abs/2111.06377) on it. The encoder
splits each window into time patches, half of them are hidden, and a throwaway
decoder is asked to reconstruct the hidden ones; the training signal comes
entirely from the recording itself, so no labels or events are used.

The difference from [`project_example`](../project_example) is what
self-supervision changes, and nothing else:

- windows are cut on a fixed `stride` across the whole recording rather than
  around events, so every sample of the recording is used;
- the segmenter has only an `"input"` extractor and no `"target"` one;
- the run ends by saving the **encoder alone**, since the decoder is
  pretraining scaffolding that downstream tasks throw away.

The example grid sweeps the two knobs that matter most for masked
pretraining: how much of the signal is hidden (`mask_ratio`) and how finely it
is cut up (`patch_size`).

## Running the example

**1. Install neuraltrain**

See the [README](../README.md) for installation instructions. The MAE encoder
needs the `models` extra:

```
pip install 'neuraltrain-repo/.[lightning,models]'
```

**2. Run local example**

```
python -m ssl_example.grids.defaults
```

This prints the path of the pretrained encoder when it finishes.

**3. Run example grid**

```
python -m ssl_example.grids.run_grid
```

**4. Evaluate the pretrained encoder**

Pretraining is only worth as much as the representations it leaves behind, so
score the encoder on a downstream `neuralbench` task by pointing `--checkpoint`
at the file from step 2:

```
neuralbench eeg audiovisual_stimulus -m mae --checkpoint <path>/encoder.ckpt
```

`neuralbench` rebuilds the same encoder, loads the weights into it, and trains
only a linear probe on top, which is what makes the score a measure of the
representations rather than of the probe. See the
[training walkthrough](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_01_train_mae.html)
for the full story.
