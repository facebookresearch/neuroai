# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- `neuralbench`: the EEG `reaction_time` task (Shirazi2024/HBN) now triggers on the contrast-change `Stimulus` instead of the `Keystroke`, so its window is 0.5 s to 2.5 s after stimulus onset as specified by the EEG Foundation Challenge 2025 (arXiv:2506.19141). The previous trigger put the whole window *after* the button press. Reported numbers for this task change and are not comparable with earlier runs (#205).
- `neuralfetch`: `Shirazi2024Hbn` contrast-change-detection trials now take `reaction_time` and `is_correct` from the first button press at or after the target onset, matching the challenge's own trial table (`eegdash.hbn.build_trial_table`). Both previously came from `groupby("trial_num").max()`, which labelled the ~8% of trials carrying several post-target presses with the last one instead of the response (0.93 s late on average, up to 2.5 s). Cached events for this study are stale: delete the `name=Shirazi2024Hbn*` folders under your `CACHE_DIR` to pick the change up (#205).
- `neuralfetch`: raised the MOABB requirement to `>=1.7.1` and removed every workaround it makes redundant — the `BNCI2020_002` reshape override, the `MartinezCagigal2023` `stim_trial` monkeypatch and the MAMEM Figshare listing cache (MOABB #1068); the `BNCI2025_001` non-EEG channel retyping, the `Cattan2019_PHMD` montage, the `Haufe2011Eeg` loader reimplementation and the `Cattan2019Passive` duration override (MOABB #1161); plus a dead `Stieger2021Continuous._get_dataset`. `Reichert2020Impact._mat_path` now asks MOABB where the file is instead of hardcoding its layout. Two behaviour changes to note: `Haufe2011Eeg` events now carry MOABB's `interval[0] = -0.5 s` shift (the p3 task's `start` moves 0.3 → 0.8 to compensate), and `Cattan2019Passive` stimulus durations become MOABB's 60 s blocks instead of ~80 s marker-to-marker spacing. Also note MOABB 1.6.0 converts `MartinezCagigal2023Checker`/`MartinezCagigal2023Pary` EEG from microvolts to volts, and `Lee2019_*` now exposes both sessions (`num_timelines` doubled and `data_shape` changed for the three `Lee2019Eeg*` studies; delete any existing `timelines.csv` for them). The `>=1.7.1` floor matters on hosts behind an HTTP proxy: up to and including 1.7.0, `get_data()` let `httpx.ProxyError` from the NEMAR sourcedata prefetch escape, so any NEMAR-registered dataset failed to load even when already on disk (MOABB #1171).
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
