# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- `neuralset`: fixed `Mne2013Sample`/`Fake2025Meg` re-downloading the MNE sample dataset on `run()` after `download()` by funneling all paths through a single root; the download progress bar now also shows when `run()` triggers the fetch. The study subfolder is now resolved once in `model_post_init` instead of mutating `self.path` in `download()`, so calling `download()` after `run()` no longer crashes on the frozen instance (#153).
- `neuralfetch`: added `Allen2022MassiveRaw` (BIDS/deepprep NSD variant) and gated NSD downloads behind the `NSD_ACCEPT_LICENCE` env var.
- `neuralbench`: added a `CLUSTER` key to `~/.neuralbench/config.json` to force fully local execution (`null`) without `--debug`, keep the default SLURM auto-detection (`"auto"`), or always submit to SLURM (`"slurm"`); honored by `--prepare`.
- `neuralbench`: a blank `WANDB_HOST` now genuinely disables Weights & Biases logging (previously `wandb.login` was still invoked).

## [0.2.1] - 2026-05-13

- `neuralset`: interactive Code Builder docs page (#39).
- `neuralset`: propagate BIDS fields to new events from transforms (#49).
- `neuralset`: fixed cache clearing logic in `Study` (#57).
- `neuralset`: fixed double-sentence issue in text transforms (#47).
- `neuralfetch`: fixed osfstorage URL in Nieuwland2018 download (#52).

## [0.2.0] - 2026-05-06

- New `neuralbench` package: unified benchmark for NeuroAI models, with
  EEG / MEG / fMRI tasks, baseline + foundation-model wrappers, plotting,
  CLI, and tutorials (#42).
- `neuralfetch`: 116 new public datasets available as `Study`
  subclasses (TUH EEG, ZuCo, ThingsMEG, EEG2Video, HBN, MOABB
  collection, …) (#41).

## [0.1.1] - 2026-05-05

- `Study.run()` fixed ProcessPool error (#37).
- `HuggingFaceText`: fixed padding for some models (#24).
- `Li2022Petit`: `Word` events now carry `language` (#30).

## [0.1.0] - 2026-04-19

- Initial release.
