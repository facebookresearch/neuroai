# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- `neuralbench`: unvalidated (`_`-prefixed) tasks get the same handcrafted sklearn baseline as their validated name (`registry.feature_based_baseline`) (#310).
- `neuralbench`: `-m fmri_mlp` keeps each fMRI task's temporal length and output head instead of imposing the image task's settings; `fmri image` configs are unchanged (#309).
- `neuralbench`: each device's defaults live in `defaults/<device>/config.yaml`, merged over `defaults/config.yaml`, and task configs no longer repeat them. `neuralbench.defaults.metrics` moves to `neuralbench.metric_configs`: task YAMLs that call `neuralbench.defaults.metrics.<fn>` must use the new path. fMRI tasks no longer build an unused channel-positions extractor (`Data.channel_positions` is optional), so their cache UIDs change (#308).
- `neuraltrain`: `DistributedClipLoss` accepts different batch sizes across ranks, so grouped batches train under DDP without `drop_last`, and the rank-based retrieval metrics sync when ranks scored different numbers of queries (#307).
- `neuralbench`: `--space` selects a task's data-representation variant (`tasks/<device>/<task>/spaces/<name>.yaml`, merged over the base config; `all` runs every space). fMRI `image` ships MNI and fsaverage spaces; models on such tasks declare `compatible_spaces` and may set `default_space`, and datasets declare `available_spaces` (#306).
- `neuralbench`: `--max-neuro-hours-per-epoch HOURS` trains each epoch on a fresh random draw of `HOURS` of training windows, cached separately from uncapped runs; `OnTheFlyPreprocessor.min_temporal_samples` zero-pads short windows for fixed-patch models such as REVE, whose channel mapping now also matches Neuromag/KIT names whatever the separator and CTF names without their serial suffix (#304).
- `neuralbench`: new LP-FT adaptation presets (`lpft_mean`, `lpft_flatten`, `lpft_attentive`) that train the head on a frozen backbone and unfreeze it at `DownstreamWrapper.unfreeze_at_epoch`, plus `finetune_attentive` and `lora_r4_attentive`; attentive-pooling runs are tagged `<strategy>_attentive`, and the bar-chart legend names each foundation model's strategy (#305).
- `neuralbench`: `-m eegnex` runs braindecode's EEGNeX (#303).
- `neuralbench`: `--plot-cached` writes under `outputs/<device>/`, and `core/` and `full/` hold one folder per foundation-model adaptation strategy (`default/` without `-w`), so the core bar chart moves from `outputs/core/core_bar_chart.png` to `outputs/eeg/core/default/core_bar_chart.png`. The figures add category leaderboards and a labelled bar chart, and replace the separate non-EEG plotting module (#302).
- docs: the NeuralBench landing page adds per-category radar plots and a customizable model scatter plot (#302).
- `neuralbench`: each device can ship a default-config overlay (`load_default_config(device)`), and `Data.drop_incomplete` drops windows missing an event for one of the extractors rather than extracting an invalid target for them (#302).
- `neuralbench`: task dataset variants are found in every plugin task root, a dataset config that replaces the default study no longer inherits that study's `query`, and the model factory accepts neuro inputs with more than one non-time dimension (#302).
- `neuralbench`: MEG `image` picks `[mag, grad]`, which leaves out the CTF reference channels that `meg` also selected on CTF systems (300 rather than 272 channels on a CTF-275) (#302).
- `neuralbench`: retrieval tasks report `test/full_retrieval/inv_norm_rank_mean` (retrieval AUC) as their headline metric rather than `top5_acc_subject-agg`, and select checkpoints on `val/batch_inv_norm_rank_mean` rather than `val/batch_top5_acc`, so scores are not comparable with earlier runs. Top-5 accuracy (`top5_acc`, `top5_acc_instance-agg`, `top5_acc_subject-agg`, `val/batch_top5_acc`) and `top1_acc_subject-agg` are still reported; the other Top-1 and Top-10 accuracies and the retrieval-rank metrics are no longer logged (#302).
- `neuralbench`: BENDR and CBraMod LoRA adapters target their whole `nn.MultiheadAttention` modules rather than their `out_proj`, which the attention forward never calls, so those LoRA runs trained no adapter. The attention inference fast path, which skips adapters, is off only inside a LoRA-wrapped model's forward. MEG tasks can run the `xdawn_ts_lr` baseline (#302).
- `neuralset`: an `MneRaw` extractor whose `filter` cutoff needs an FIR design longer than the recording's longest contiguous segment skips that cutoff with a warning and applies the other one (`on_filter_too_long="drop"`; `"raise"` rejects the recording), rather than applying a filter dominated by edge artifacts. Under the EEG tasks' `[0.1, 75]` Hz filter this drops the high-pass on all 1,260 `Cattan2019Dataset` recordings, 320 of 800 `Sosulski2019Electroencephalogram` ones and 2 of 518 `Harati2015Tuev` ones, so their scores are not comparable with earlier runs (#302).
- `neuralset`: `FmriCleaner` raises when `high_pass` or `low_pass` is set without a `filter`, which silently dropped the band-pass; `TimeAggregatedExtractor`, `CroppedExtractor` and `ToStatic` subclass `BaseExtractor` rather than `BaseStatic`, so `isinstance(x, BaseStatic)` no longer matches them (#302).
- `neuraltrain`: LoRA adapters reach braindecode's fused-qkv attention layers, which bypassed them; LaBraM selects channels from its known channel names alone, ignoring per-sample `channel_positions`, so the channel count it forwards does not depend on the batch; REVE channel handling fixes (#302).
- `neuraltrain`: `PearsonCorrCoef` accumulates in float64 (float32 on MPS, which has no float64), so it no longer returns NaN on low-variance targets; new `InverseNormalizedRank` retrieval metric and `DistributedClipLoss`, the CLIP loss over the global batch under DDP; `fit()` no longer crashes on CPU after an explicit `trainer.validate()` (#302).
- `neuralfetch`: `Singh2021Timing` no longer leaks between splits, and its version bump rebuilds cached timelines; `Harati2015Tuev` merges an event's per-channel annotations even when their durations differ, rather than splitting it into several events (#302).
- `neuralset`: extractors batch their per-segment outputs through a `collate()` method (default `torch.stack`) that `SegmentDataset` calls, and `load_all` takes a `batch_size`; neuro extractors accept `store_dtype="float16"` to halve their cache; image and video extractors keep their outputs in float32 (#302).
- `neuralset`: new `Ecg` event and `EcgExtractor`; a failed study lookup suggests close matches (#302).
- `neuralbench`: `data.stream_by` names the event fields that identify an evaluation stream. The model is reset at each new stream (`ResetPerStream`), and `GroupedMetric` metrics score each stream separately, then average. `_motor_imagery_stream` streams each subject's session, its runs in recording order, and `_sleep_onset_stream` each recording (#301).
- `neuralbench`: `BinnedMAE` takes `bin_weights` for the weighted W-bMAE, and `GroupedMetric` finds metrics defined in any package. `_sleep_onset_stream` reports and selects checkpoints on `wbmae_stream_mean` (weights 10 / 5 / 3 / 1, per recording, then averaged over recordings), and `_motor_imagery_stream` on `bal_acc_stream_mean` (balanced accuracy per session, then averaged) (#301).
- `neuralbench`: the stream tasks preprocess as the EEG/EMG Foundation Challenge 2026 does. `_motor_imagery_stream` keeps the 120 Hz resampling and the filters but drops the scaler and clamping, and feeds microvolts; `_sleep_onset_stream` feeds the signal at its native rate, unfiltered, in microvolts. Model configs with their own preprocessing, such as `reve`, override this; a streamed test warns when `data.neuro.scaler` is set, as it is fit on whole recordings. Muse recordings are no longer cropped to 1,200 s before N2 or started at random, in `_sleep_onset_stream` or `sleep_onset --dataset interaxon2026muse`, and both use the supplied session split instead of the subject-disjoint one, validating on subjects without a test recording so every test sleeper keeps its training data; scores are not comparable with earlier runs (#301).
- `neuralfetch`: `Dreyer2026Proteus` lists each session's runs in recording order rather than by task name (#301).
- `neuralfetch`/`neuralbench`: new `Interaxon2026Muse` study, the public training set of the EEG/EMG Foundation Challenge 2026 Track 3 corpus (NEMAR nm000287 v1.0.0, ~1.1 GB: 540 at-home recordings from 203 participants, 4 Muse EEG channels at 128 Hz, one first-N2 annotation per recording), and its `interaxon2026muse` variant of `eeg sleep_onset`, on the supplied session split (500 train / 40 seen-participant test recordings); the study keeps the supplied train / test session labels as `split` (#283).
- `neuralfetch`/`neuralbench`: new `Dreyer2026Proteus` study, the training release of the EEG/EMG Foundation Challenge 2026 Track 2 corpus (NEMAR nm000290 v1.0.0, ~1.5 GB: 112 runs from 10 participants, 41 EEG channels at 500 Hz), and its `dreyer2026proteus` variant of `eeg motor_imagery`: mi / sub / word cues, 1-5 s after cue onset, on the task's subject-disjoint split (#294).
- `neuralbench`: added the `eeg _motor_imagery_stream` task for Track 2 of the EEG/EMG Foundation Challenge 2026: `motor_imagery --dataset dreyer2026proteus` with its test split scored one window at a time, in time order, and the model reset at each new session (`data.test_batch_size: 1`, `data.stream_by: [subject, session]`). Its other dataset variants are the two other corpora the Track 2 docs use, `dreyer2023` and `tangermann2012`; those docs use it throughout (#299).
- `neuralbench`: `eeg _sleep_onset_stream` defaults to `Interaxon2026Muse` rather than Sleep-EDF, which `--dataset kemp2000analysis` selects, so a run without `--dataset` is not comparable with earlier ones. The Track 3 docs build on the new default, the Codabench warm-up set (#283).
- `neuralbench`: `emg pose` feeds EMG to the model in microvolts (`scale_factor: 1.0e+6`), the raw HDF5 scale emg2pose trains on, rather than the volts `mne` returns — at volt scale the signal variance sits far below the epsilon of `VEMG2Pose`'s first `LayerNorm`, which flattened the input. Scores are not comparable with earlier runs; the prepared EMG cache is reused as is (#296).
- docs: the challenge overview's `VEMG2Pose` baseline is 16.46 +/- 0.17 degrees `test/mae` over three seeds, rerun after the microvolt fix; the 25.14 +/- 2.30 it replaces was the `emg typing` character error rate, carried over when the column became EMG pose (#296).
- `neuralfetch`: `Zyma2019Electroencephalograms` reads all 36 EEGMAT subjects, including `Subject00`, which it skipped (#289). The `eeg mental_arithmetic` subject split changes with it (1,707 windows over 36 subjects, 1,007 for training, against 1,659 over 35), so scores are not comparable with earlier runs. A study cache written before the fix still holds 35 subjects and is reused as is: delete `<CACHE_DIR>/name=Zyma2019Electroencephalograms-*` to pick up the new subject.
- `neuralfetch`: new `Nemar` download backend (nemar-py; `version=` pins the release, `include`/`exclude` select files); `Xu2024Alljoined` and `Xu2025Alljoined` use it for the NEMAR BIDS releases nm000133 v1.0.4 and nm000134 v1.0.3 (~5 and ~8.5 GB), with images from each event's `stim_file` (#279). Copies fetched from OSF or Hugging Face are not read: re-run with `--download`. Alljoined-1 keeps every presented trial (49,541 over 13 recordings); the OSF version read the same raw EEG but took its trial list from the `05_125` epoch files, which omit the trials AutoReject rejected (43,070 over 12), so scores are not comparable with earlier runs; Alljoined-1.6M's image events are unchanged except for 49 test trials that shared an onset with a `debug` marker and are kept.
- `neuralfetch`: `Kemp2000Analysis` downloads Sleep-EDF from PhysioNet's `physionet-open` S3 mirror, as the other PhysioNet studies already do, instead of the `physionet.org` web host, which served it at 51 KB/s against the mirror's 31 MB/s on the host we measured — minutes for the ~7 GB corpus rather than tens of hours. Recordings now live under `download/sleep-edfx/1.0.0/sleep-cassette/`, and the `physionet-sleep-data/` folder the previous `mne` fetch wrote is still read where it lies, so a copy fetched before the switch is neither re-downloaded nor silently invisible.
- `neuralfetch`: `success_writer` clears the marker before an `overwrite` re-run rather than leaving it on disk, so a re-run that dies partway no longer looks complete to the next call, which previously skipped the work and accepted half-written output (#278).
- docs: REVE's `brain-bzh` checkpoints are public, so the starter kit, `install.md` and the model page no longer ask for a HuggingFace account, an accepted licence or a token.
- docs: each EEG/EMG Foundation Challenge 2026 track page opens with a cheaper way in than its default corpus — `xu2024alljoined` for Track 1, `tangermann2012` for Track 2, and `--download --debug` for Track 4.
- docs: EEG/EMG Foundation Challenge 2026 starter-kit corrections from participant feedback — every track page documents its exact split and the validation metric that selects the final checkpoint, `--dataset dreyer2023` is presented as Track 2's recommended warm-up configuration (the task default is still `Stieger2021Continuous`), and Track 3's target is the first N2 epoch rather than the first *stable* one, on a 128 Hz cohort holding both seen and unseen sleepers (#277).
- docs: the challenge track pages separate warm-up from sealed specifications, which differ for Tracks 1-3 — Track 3 scores unweighted bMAE on the Sleep-EDF warm-up against the sealed W-bMAE (10x/5x/3x/1x severity weights, macro-averaged over seen and unseen subjects), and Track 2 pools balanced accuracy over windows where the sealed phase averages over subject-session-context cells. Track 4's split section no longer claims held-out users: its `user_stage` test set holds unseen user-stage *combinations* (#277).
- docs: the challenge submission page defers the submission contract, registration and benchopt workflow to Codabench, which owns and updates them, and keeps only what NeuralBench owns — getting weights out of a run (`delete_checkpoints_on_exit`, stripping the `pl_module` prefix), the preprocessing and unit gaps a lifted model has to close, and whether a local score tracks the leaderboard (#277).
- docs: EEG/EMG Foundation Challenge 2026 starter-kit corrections from the post-release audit — the pretraining corpus needs an explicit `ssl_example.grids.download` step and `moabb`, `MaeModule` takes no other `neuraltrain` model, freezing comes from `linear_probe_mean` rather than `mae.yaml`, `install.md` lists the six config keys that have no default, Track 3's target is the earliest annotated N2 event, and emg2pose is ~340 GB (#251).

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
