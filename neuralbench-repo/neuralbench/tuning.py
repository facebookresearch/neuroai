# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive hyperparameter search for ``neuralbench --optuna``.

Each assembled experiment config gets an Optuna study, stored in a journal file
under ``<storage_dir>``.  Trials are asked in batches, run as ordinary
experiments (so they are submitted, parallelised and cached like any benchmark
run), and told their best validation score, the config's
``trainer_config.monitor``.  The test set never steers the search.

Each run reports its validation monitor to its trial every epoch through
optuna-integration's ``PyTorchLightningPruningCallback``, which reaches the trial
in the journal file from the training job; a median pruner stops unpromising
trials early.  Re-running the same command resumes the studies.
"""

import hashlib
import logging
import typing as tp
from pathlib import Path

import pydantic
import yaml
from exca import ConfDict

from neuralbench.registry import DEFAULTS_DIR, _resolve_task_dir

LOGGER = logging.getLogger(__name__)

RunBatch = tp.Callable[[list[ConfDict]], list[dict[str, tp.Any] | None]]


class OptunaTrial(pydantic.BaseModel):
    """The Optuna trial an experiment reports its validation monitor to."""

    model_config = pydantic.ConfigDict(extra="forbid")

    storage: str  # journal file
    study_name: str
    trial_id: int

    def pruning_callback(self, monitor: str) -> tp.Any:
        import optuna
        from optuna_integration import PyTorchLightningPruningCallback

        # the pruner is not stored with the study, so the job builds the same one
        study = optuna.load_study(
            study_name=self.study_name, storage=_storage(self.storage), pruner=_pruner()
        )
        return PyTorchLightningPruningCallback(
            optuna.trial.Trial(study, self.trial_id), monitor
        )

    @staticmethod
    def pruned(error: BaseException) -> bool:
        import optuna

        return isinstance(error, optuna.TrialPruned)


def _pruner() -> tp.Any:
    import optuna

    # prunes a trial below the median of the completed ones at the same epoch,
    # once 5 trials have completed
    return optuna.pruners.MedianPruner()


def _storage(path: str | Path) -> tp.Any:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(path)))


def load_search_space(device: str, task_name: str) -> dict[str, tp.Any]:
    """``defaults/search_space.yaml``, with the task's ``search_space.yaml`` keys over it.

    Keys are dotted config paths; a list is a categorical choice and a
    ``{low, high, log, step}`` mapping a float range (an int range when both
    bounds are ints).
    """
    space = _read(DEFAULTS_DIR / "search_space.yaml")
    space.update(_read(_resolve_task_dir(device, task_name) / "search_space.yaml"))
    return space


def _read(path: Path) -> dict[str, tp.Any]:
    if not path.exists():
        return {}
    # plain yaml: ConfDict would split the dotted keys into nested dicts
    return yaml.safe_load(path.read_text()) or {}


def _distribution(spec: tp.Any) -> tp.Any:
    import optuna.distributions as d

    if isinstance(spec, list):
        return d.CategoricalDistribution(spec)
    if isinstance(spec["low"], int) and isinstance(spec["high"], int):
        return d.IntDistribution(**spec)
    return d.FloatDistribution(**spec)


def tune(
    configs: list[ConfDict],
    space: dict[str, tp.Any],
    run_batch: RunBatch,
    n_trials: int,
    storage_dir: str | Path,
    batch_size: int = 1,
    seed: int = 0,
) -> list[tp.Any]:
    """Run each config's study up to *n_trials* finished trials, *batch_size* at a time.

    *run_batch* runs trial configs and returns their result dicts (``None`` for
    a failed run).  Returns the studies.
    """
    import optuna

    distributions = {key: _distribution(spec) for key, spec in space.items()}
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    studies = []
    for config in configs:
        name = config.to_uid()
        path = (
            Path(storage_dir)
            / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.journal"
        )
        study = optuna.create_study(
            study_name=name,
            storage=_storage(path),
            load_if_exists=True,
            direction="maximize"
            if config["trainer_config.mode"] == "max"
            else "minimize",
            pruner=_pruner(),
        )
        # reseeded with the trial count, so a resumed study does not redraw its trials;
        # constant_liar spreads the trials asked before a batch's results are in
        study.sampler = optuna.samplers.TPESampler(
            seed=seed + len(study.trials), constant_liar=True
        )
        studies.append((study, config, str(path)))
    running = optuna.trial.TrialState.RUNNING
    while True:
        asked = []
        for study, config, journal in studies:
            # trials an interrupted run left running go first: their finished
            # experiments come straight from the cache
            trials = [
                optuna.trial.Trial(study, t._trial_id)
                for t in study.trials
                if t.state == running
            ]
            n_finished = len(study.trials) - len(trials)
            n_new = min(batch_size - len(trials), n_trials - n_finished - len(trials))
            trials += [study.ask(distributions) for _ in range(max(n_new, 0))]
            asked += [(study, config, journal, trial) for trial in trials]
        if not asked:
            break
        trial_configs = []
        for study, config, journal, trial in asked:
            trial_config = config.copy()
            trial_config.update(trial.params)
            trial_config["optuna_trial"] = OptunaTrial(
                storage=journal, study_name=study.study_name, trial_id=trial._trial_id
            ).model_dump()
            trial_configs.append(trial_config)
        for (study, config, _, trial), result in zip(asked, run_batch(trial_configs)):
            monitor = config["trainer_config.monitor"]
            if result is not None and result.get("optuna_pruned"):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            elif result is None or monitor not in result:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, result[monitor])
    for study, config, _ in studies:
        name = f"{config['task_name']} / {config.get('brain_model_name', '?')}"
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            # ready to paste into a --config-overlay file
            LOGGER.info(
                "Best %s for %s: %.4f with\n%s",
                config["trainer_config.monitor"],
                name,
                study.best_value,
                yaml.safe_dump(study.best_params, sort_keys=False).rstrip(),
            )
        else:
            LOGGER.warning("No Optuna trial completed for %s", name)
    return [study for study, _, _ in studies]
