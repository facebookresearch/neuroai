# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- docs: the theme's "View this page" and "Edit this page" buttons on a sphinx-gallery page point at the example they are built from, rather than at the rst sphinx-gallery generates at build time and never commits, which 404'd on all 42 gallery pages (#252).

## [0.3.1] - 2026-09-10

- `neuralbench`: adaptation wrappers (`-w`) apply to every model whose config ships a `downstream_model_wrapper`, not just the published foundation models, so a locally pretrained `mae` encoder is linear-probed as documented instead of silently fine-tuned end to end (#249).
- `neuralbench`: the GPU capability check warns rather than aborting when the driver is too old for the installed torch, so `--download`, `--prepare` and `--plot-cached` still run on such a host (#249).
- `neuralbench`: `load_config` creates the directories a hand-written `config.json` names, and reports the `/tmp` fallback it takes when no config exists and stdin is not a terminal (#249).
- `neuralfetch`: a MOABB study whose subjects all fail to download raises an error naming the counts and chaining the underlying failure, instead of asserting that `timelines.csv` is missing (#249).
- `neuralset`: `ChannelPositions` names the montage it resolved against, and the fix, when no channel has a valid position (#249).
- docs: EEG/EMG Foundation Challenge 2026 starter-kit corrections — Track 1's headline metric key, Track 3's per-window target, the Track 4 aggregation command, the dataset clients and download footprints each track needs, `ssl_example`'s clone step and its own cache paths, and the Codabench submission route (#249).

## [0.3.0] - 2026-09-09

- `neuralbench`: added the `emg pose` task — 20-joint hand-angle trajectory regression on EMG2Pose (NEMAR NM000281), in the paper's regression setting (#229).
- `neuralfetch`: read EMG2Pose paper split metadata from the BIDS `scans.tsv`, falling back to the upstream `emg2pose_metadata.csv` on NEMAR releases that omit those columns (#229).
- `neuralbench`: `Data.min_finite_target_fraction` drops windows labelled over less than that fraction of their frames. `emg pose` sets 1.0 to match emg2pose's `skip_ik_failures`; a lower value trades that parity for data (#229).
- `neuralbench`: `emg pose` tests on EMG2Pose's held-out user+stage set alone, the paper's hardest scenario; the paper scores its three test sets separately, so a pooled score matches none of them. `val` keeps both scenarios (#229).
- `neuralbench`: multi-GPU runs reach the test phase — reading the best checkpoint to report `best_epoch` and deleting it on exit are rank-zero only, so the other ranks no longer fail on a checkpoint they never wrote or strip it before rank zero tests with it (#229).
- `neuraltrain`/`neuralbench`: `BandRotationConfig` wraps braindecode's `BandRotation`, and `Experiment.augmentation` applies an augmentation to the neuro input during training only. `emg pose` rotates its band by one position (#229).
- `neuralbench`: `emg pose` trains on joint angles in radians, as emg2pose does, rather than degrees — scaling the target inflated gradients and left `VEMG2Pose` unable to fit. Reported metrics are radians; multiply by 57.29578 to compare with the paper (#229).
- `neuralbench`: the `emg pose` extractors read from the disk cache instead of also holding recordings in RAM — exca's RAM cache never evicts, so a full run grew past 250 GB and was OOM-killed (#229).
- `neuralbench`: `test/mae` is the headline metric for `L1Loss`, so `--plot-cached` aggregates regression tasks such as `emg pose` instead of raising `KeyError` (#229).
- `neuralbench`: the EEG `reaction_time` task triggers on the contrast-change `Stimulus`, not the `Keystroke`, so its window is 0.5–2.5 s after stimulus onset per the EEG Foundation Challenge 2025 (arXiv:2506.19141). Numbers are not comparable with earlier runs (#234).
- `neuralfetch`: `Shirazi2024Hbn` takes `reaction_time` and `is_correct` from the first button press at or after target onset rather than the last, which mislabelled ~8% of trials. Delete the `name=Shirazi2024Hbn*` folders under your `CACHE_DIR` (#234).
- `neuralfetch`: raised the MOABB floor to `>=1.7.1`, dropping the workarounds it makes redundant (MOABB #1068, #1161) and fixing NEMAR dataset loads behind an HTTP proxy (MOABB #1171) (#226).
- `neuralfetch`: MOABB 1.7.1 changes data — `Haufe2011Eeg` events shift by -0.5 s, `Cattan2019Passive` durations become 60 s blocks, `MartinezCagigal2023*` rescales to volts, and `Lee2019Eeg*` exposes both sessions, so delete their `timelines.csv` (#226).
- `neuralset`: `Study._body()` resolves `timelines.infra` on use rather than at construction, so a `Study` inside a `Chain` keeps its per-timeline backend instead of silently loading timelines sequentially and uncached (#240).
- `neuralset`: `HuggingFacePCA` stages under a deterministic folder, clears it once the PCA is cached, and releases the extractor cache in a `finally`. Killed runs used to leave unbounded `HF-PCA-tmp*` folders and open fds over NFS (#239).
- `neuralset`: added the `ClipVersatileDiffusion` image extractor (#246).
- `neuralset`: `HuggingFaceMixin` accepts `token_aggregation="cat"` (#245).
- `neuralset`: `Segmenter.padding` accepts `"auto"`, padding segments to `max(segments.duration)` as `SegmentDataset.pad_duration` already did (#244).
- `neuralset`: `AggregatedExtractor` accepts static extractors configured with a non-zero frequency — streams are classified by frequency rather than by inheritance — and takes the common frequency as its own (#244).
- `neuralset`: lightweight extractors wrapping a Slurm extractor (such as `ChannelPositions` around `MneRaw`) are prepared once the futures finish, instead of recursively launching duplicate work for the same cache uid (#244).
- `neuralset`: `channel_order` is excluded from `MneRaw`'s cache uid, so reordering channels reuses the cached extraction (#244).
- `neuralset`: fixed the channel-mapping docstring to match the current dimensions (#232).
- `neuraltrain`: added an `ssl_example` MAE pretraining project and a training docs page, for the EEG/EMG Foundation Challenge 2026 starter kit (#237).
- `neuraltrain`: `neuraltrain.models` exports `BaseBrainModelConfig` and `BrainModelBuildContext`, and `ChannelMerger` is a `BaseBrainModelConfig` (#244).
- `neuraltrain`: W&B login runs on rank zero only. Under Slurm/srun every rank called `wandb.login`, and concurrent attempts to start the local authentication service could race and fail before Lightning's rank-zero guard applied (#244).
- `neuralfetch`: added a `Gin` download backend for gin.g-node.org, which registers annex URLs and pulls with `git annex get --from=web` — plain `datalad get` leaves only pointer files there. Same arguments as `Datalad`, plus `branch=` (#244).
- `neuralfetch`: `Figshare` verifies each file's MD5, downloads through a `.part` file, and retries on mismatch. `skip_existing` re-checks files already on disk, repairing datasets that an expired URL had corrupted with an error page (#238).
- `neuralfetch`: every `Study._download` declares `overwrite`, so `neuralfetch download <Study>` behaves uniformly across studies (#222).
- `neuralfetch`: OpenNeuro downloads with `overwrite=False` skip a target that already holds data instead of letting openneuro-py re-verify and repair it in place (#224).
- `neuralbench`: adaptation strategies for foundation models — linear probe, attentive probe, LoRA and full fine-tuning — selectable per model, with adaptation-aware plots and tables. Requires `peft>=0.13` and `transformers>=4.43` (#230).
- `neuralbench`: added a Bring-Your-Own-Model evaluation API, with an `03_evaluate_your_own_model` quickstart tutorial; `labram` and `reve` are reworked onto it (#241).
- `neuralbench`: `get_default_dataloaders` returns a task's dataloaders standalone, without running an experiment. Accepts dataset variants and rejects unknown datasets and task aggregates (#209, #228).
- `neuralbench`: dropped the `torch==2.6` pin for `torch>=2.5.1`, matching `neuralset` and `neuraltrain`. The CLI warns instead when a visible GPU's CUDA capability is absent from the installed torch's arch list, naming the reinstall that fixes it (#236).
- `neuralbench`: depends on `neuralfetch[quickstart]` rather than carrying its own `datasets` extra, so the study catalog and its download backends arrive together (#233).
- `neuralbench`: `ModelEntry` carries backbone and pretraining-corpus descriptions (#242).
- `neuralbench`: `RegressionBinSampler` edge binning matches `BinnedMAE` (#183).
- `neuralbench`: task configs take the cluster from `~/.neuralbench/config.json` instead of hardcoding it (#225).
- `neuralbench`: dropped competition track 5 (foundation-model transfer) from the Biosignal Challenge 2026 starter kit (#227).
- docs: the NeuralSet paper is cited as arXiv:2605.03169 rather than a preprint PDF (#197).

## [0.2.3] - 2026-07-31

- `neuralset`: `Study.version` is now a top-level field; `infra_timelines` → `timelines.infra` (Step syntax, defaults to `ProcessPool`). Requires exca ≥ 0.5.27. (#194)
- `neuralset`: fMRI ROI support — `CiftiRoiProjector` for cortical Glasser and subcortical ROIs, Glasser fsaverage ROI projection, and ROI query/selection options (#185, #210, #211, #215).
- `neuralset`: support for Algonauts CIFTI fMRI derivatives (#191).
- `neuralset`: unified HuggingFace model loading, refactored the visual extractors, and added model prefetching (#102, #135, #145, #151).
- `neuralset`: `HuggingFaceText` truncates context to the model's positional capacity (#121).
- `neuralset`: added `TimedArray.copy(**changes)` (#206).
- `neuralset`: added an `on_trigger_overlap` guard to `list_segments` (#115).
- `neuralset`: `ensure_finite` is honored when other cleaning is disabled (#164).
- `neuralset`: duration calculation accounts for precision (#138).
- `neuralset`: fixed `Mne2013Sample`/`Fake2025Meg` re-downloading MNE sample data on `run()` after `download()` (#157).
- `neuraltrain`: declared `exca` as a runtime dependency (#156).
- `neuralfetch`: added `Allen2022MassiveRaw` (BIDS/deepprep NSD variant) and gated NSD downloads behind `NSD_ACCEPT_LICENCE` (#105).
- `neuralfetch`: added the `Levy2026Noninvasive` (SpanishBCBL/DECOMEG) typing study (#201).
- `neuralfetch`: combined the THINGS fMRI and MEG studies into `hebart2023things` (#131).
- `neuralfetch`: aligned study file names with their class names (#119, #186).
- `neuralfetch`: fixed `Brennan2019Hierarchical` loading for the v2 (bn999738r) release (#175).
- `neuralbench`: added the EMG `qwerty` CTC keystroke-decoding task (#50).
- `neuralbench`: added a `CLUSTER` key to `~/.neuralbench/config.json` (`null` = local, `"auto"` = SLURM auto-detect, `"slurm"` = always SLURM); honored by `--prepare` (#118).
- `neuralbench`: blank `WANDB_HOST` now disables W&B logging (previously `wandb.login` was still called) (#118).
- `neuralbench`: prediction logging (#126).
- `neuralbench`: added a `probe_layer` field to `DownstreamWrapper` for layer-wise linear probing (#83).
- `neuralbench`: support for the `build_from_context` brain-model build API (#207).

## [0.2.2] - 2026-05-26

- `neuralset`: extended `ChunkEvents` to fMRI and `MneRaw` inputs (#81).
- `neuralset`: single aggregation allows non-overlapping events (#95).
- `neuraltrain`: `dilation_growth` accepts floats (#80).
- `neuralfetch`: `StudyInfo`-powered dataset explorer in the docs (#76, #78, #91).
- `neuralfetch`: switched the Excel engine from openpyxl to calamine (~6× faster) in the Nieuwland and Chen studies (#69).
- `neuralfetch`: replaced deprecated `pick_types()` with `pick()` in `Nieuwland2018Large` (#68).
- `neuralfetch`: fixed the digest type for SHA-256 in `download.py` (#85).
- `neuralbench`: added the sleep-onset prediction task (#89).
- `neuralbench`: refactored RNG handling in `Data`, and seeded before data setup and model construction (#70, #74).

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
