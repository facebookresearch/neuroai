# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
