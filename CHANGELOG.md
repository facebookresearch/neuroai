# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- `neuralset`: `Study.version` is now a top-level field; `infra_timelines` → `timelines.infra` (Step syntax, defaults to `ProcessPool`). Requires exca ≥ 0.5.27. (#194)
- `neuralset`: fixed `Mne2013Sample`/`Fake2025Meg` re-downloading MNE sample data on `run()` after `download()` (#157).
- `neuralfetch`: added `Allen2022MassiveRaw` (BIDS/deepprep NSD variant) and gated NSD downloads behind `NSD_ACCEPT_LICENCE` (#105).
- `neuralbench`: added `CLUSTER` key to `~/.neuralbench/config.json` (`null` = local, `"auto"` = SLURM auto-detect, `"slurm"` = always SLURM); honored by `--prepare` (#118).
- `neuralbench`: blank `WANDB_HOST` now disables W&B logging (previously `wandb.login` was still called) (#118).

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
