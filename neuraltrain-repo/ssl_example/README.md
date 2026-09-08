# `neuraltrain` self-supervised example

This directory contains an example project showing how to use `neuraltrain` and
`pytorch-lightning` to pretrain an encoder by masked prediction on unlabelled
EEG data, and how to hand that encoder to `neuralbench` for evaluation.

## Description

This example pools the EEG datasets behind challenge tracks 1-3
(`Gifford2022Large`, `Stieger2021Continuous`, `Kemp2000Analysis`) with one
resting-state dataset that belongs to no track (`Miltiadous2023Dice`), and
pretrains a small [MAE](https://arxiv.org/abs/2111.06377)-style encoder on all
of them at once. Each window is split into time patches, half of them are
replaced by a learned mask token, and a single linear layer reconstructs the
hidden ones from the encoder's output; the training signal comes entirely from
the recordings themselves, so no labels or events are used.

Those four datasets share no montage — they range from a 63-channel cap to
Sleep-EDF's two bipolar derivations — so the encoder starts with a
`ChannelMerger`, which maps whatever channels a recording has onto a fixed set
of virtual ones using their 3D positions. Channels a recording does not have
are zero-padded by the extractor, arrive with invalid positions, and are masked
out of the merge. That is also what lets one checkpoint score on a downstream
task with a montage of its own.

The model is **encoder-only**: the original MAE encodes just the visible patches
and restores the rest with a transformer decoder, which is cheaper per step and
a natural first thing to try improving here.

The difference from [`project_example`](../project_example) is what
self-supervision changes, and nothing else:

- several studies are pooled into one events table, rather than one study;
- windows are cut on a fixed `stride` across the whole recording rather than
  around events, so every sample of the recording is used;
- the segmenter has an `"input"` extractor and channel positions, but no
  `"target"` one;
- the run ends by saving the **encoder alone**, since the mask token and the
  reconstruction layer are pretraining scaffolding that downstream tasks throw
  away.

The example grid sweeps the knob that matters most for masked pretraining: how
much of the signal is hidden (`mask_ratio`).

## Running the example

**1. Install neuraltrain**

See the [README](../README.md) for installation instructions. The encoder needs
the `models` extra:

```
pip install 'neuraltrain-repo/.[lightning,models]'
```

**2. Check the wiring**

The four datasets are large, so start with the debug config, which swaps them
for one small bundled recording and runs a single batch:

```
python -m ssl_example.grids.test_run
```

**3. Run the real thing**

```
python -m ssl_example.grids.defaults
```

This downloads the four datasets on first use and prints the path of the
pretrained encoder when it finishes. Set `wandb_config` to `None` in
`grids/defaults.py` to train without Weights & Biases. To pretrain on fewer
datasets, or on your own, edit `STUDIES` in the same file.

**4. Run example grid**

```
python -m ssl_example.grids.run_grid
```

**5. Evaluate the pretrained encoder**

Pretraining is only worth as much as the representations it leaves behind, so
score the encoder on a downstream `neuralbench` task by pointing `--checkpoint`
at the file from step 3:

```
neuralbench eeg motor_imagery -m mae --checkpoint <path>/encoder.ckpt
```

`neuralbench` rebuilds the same encoder, loads the weights into it, freezes it,
and trains only a linear probe on top, which is what makes the score a measure
of the representations rather than of the probe. Note that pretraining above
includes the datasets of tracks 1-3, so a score on one of those tasks says
nothing about generalising to unseen data. See the
[training walkthrough](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html)
for the full story, including how to keep `mae.yaml` in step with your
pretraining config.
