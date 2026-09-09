# `neuraltrain` self-supervised example

This directory contains an example project showing how to use `neuraltrain` and
`pytorch-lightning` to pretrain an encoder by masked prediction on unlabelled
EEG data, and how to hand that encoder to `neuralbench` for evaluation.

## Description

The example pools four public EEG datasets -- image viewing
(`Gifford2022Large`), motor imagery (`Stieger2021Continuous`), sleep
(`Kemp2000Analysis`) and resting state (`Miltiadous2023Dice`), together some 240
subjects -- and pretrains a small [MAE](https://arxiv.org/abs/2111.06377)-style
encoder on all of them at once. Each window is split into time patches, half of
them are replaced by a learned mask token, and a single linear layer
reconstructs the hidden ones from the encoder's output. The loss is a plain MSE
over the hidden patches only; scoring the visible ones too would reward copying
the input. No labels or events are used, so the training signal comes entirely
from the recordings themselves.

Those four datasets share no montage -- they range from a 63-channel cap to
Sleep-EDF's two bipolar derivations -- so a channel is never identified by its
index. One token is one channel over one time patch, and it carries a Fourier
embedding of that channel's 3D position on the head alongside the embedding of
its time patch. Channels a recording does not have are zero-padded by the
extractor, arrive with invalid positions, and have their tokens dropped from
the attention and excluded from the reconstruction targets. That is also what
lets one checkpoint score on a downstream task with a montage of its own; the
cost is a sequence of `n_channels * n_patches` tokens.

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
- the split holds out whole subjects, so the validation loss measures
  reconstruction of a recording the encoder has never seen rather than of
  another window of one it trained on;
- the run ends by saving the **encoder alone**, since the mask token and the
  reconstruction layer are pretraining scaffolding that downstream tasks throw
  away.

The example grid sweeps the knob that matters most for masked pretraining: how
much of the signal is hidden (`mask_ratio`).

Preprocessing (sampling rate, filters, scaling) is the set `neuralbench` applies
by default to EEG tasks, so the encoder sees the same signal during pretraining
as it will during evaluation.

## Running the example

**1. Install neuraltrain**

See the [README](../README.md) for installation instructions. The encoder needs
the `models` extra:

```
pip install 'neuraltrain-repo/.[lightning,models]'
```

**2. Check the wiring**

The four datasets are downloaded on first use and then preprocessed into a
cache, which takes a while, so start with the debug config. It swaps them for
one small bundled recording and runs a single batch:

```
python -m ssl_example.grids.test_run
```

**3. Run the real thing**

```
python -m ssl_example.grids.defaults
```

This downloads the four datasets on first use, caches their preprocessed form,
and prints the path of the pretrained encoder when it finishes. Set
`wandb_config` to `None` in `grids/defaults.py` to train without Weights &
Biases. To pretrain on fewer datasets, or on your own, edit `STUDIES` in the
same file.

**4. Run example grid**

```
python -m ssl_example.grids.run_grid
```

**5. Evaluate the pretrained encoder**

Pretraining is only worth as much as the representations it leaves behind, so
score the encoder on a downstream `neuralbench` task by pointing `--checkpoint`
at the file from step 3:

```
neuralbench eeg motor_imagery -m mae --checkpoint <path>/encoder.ckpt \
    -w linear_probe_mean
```

`neuralbench` rebuilds the same encoder and loads the weights into it, and
`-w linear_probe_mean` freezes it and trains only a linear probe on top, which
is what makes the score a measure of the pretrained representation rather than
of the fine-tuning that would otherwise follow. Note that pretraining above includes the dataset behind this
task, so the score says nothing about generalising to data the encoder has never
seen -- use a task built on another dataset for that.

## Using this example for the biosignal challenge

This example doubles as the starter kit for the training stage of the
[EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/). The
[training walkthrough](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html)
covers the same ground with the competition in view, including which datasets
back which track, how to keep `mae.yaml` in step with your pretraining config,
and how to submit a model trained outside this repository.
