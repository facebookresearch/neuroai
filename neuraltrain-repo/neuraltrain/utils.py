# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility scripts."""

from __future__ import annotations

import collections
import inspect
import pickle
import shutil
import time
import typing as tp
from itertools import product
from math import prod
from pathlib import Path
from warnings import warn

import exca
import numpy as np
import pydantic
import submitit
import torch
from pydantic import Field, create_model

from neuraltrain.base import BaseModel


def convert_to_pydantic(
    class_to_convert: type,
    name: str,
    parent_class: tp.Any = None,
    exclude_from_build: list[str] | None = None,
) -> pydantic.BaseModel:
    """
    Converts any class into a pydantic BaseModel. Initialize the class
    with the 'self.build()' method

    If parent_class inherits from exca.helpers.DiscriminatedModel, the name
    field is not added as it's handled automatically by DiscriminatedModel.
    """
    # Get the constructor of the class
    init = class_to_convert.__init__  # type: ignore

    # Inspect signature
    sig = inspect.signature(init)
    empty = inspect.Parameter.empty

    if "name" in sig.parameters:
        raise RuntimeError("Cannot convert class with attribute 'name' to pydantic")

    fields = {
        k: (
            v.annotation if v.annotation != empty else tp.Any,
            v.default if v.default != empty else ...,
        )
        for k, v in sig.parameters.items()
        if k != "self" and not k.startswith("_")
    }

    # add name for pydantic.discriminator (unless using DiscriminatedModel)
    fields["name"] = (tp.Literal[name], Field(default=name))
    # Check if parent uses DiscriminatedModel (which handles 'name' automatically)
    if parent_class is not None:
        if issubclass(parent_class, exca.helpers.DiscriminatedModel):
            del fields["name"]  # must not be added anymore

    # Create the Pydantic model class dynamically
    Builder = create_model(  # type: ignore
        name,
        __base__=parent_class,
        **fields,
    )
    Builder._cls = class_to_convert  # type: ignore

    # Define a build method to instantiate the original class
    if exclude_from_build is None:
        exclude_from_build = []

    def build_method(instance: BaseModel):
        params = dict(
            (field, getattr(instance, field))
            for field in type(instance).model_fields
            if (field != "name" and field not in exclude_from_build)
        )
        return instance._cls(**params)  # type: ignore

    # Bind the build method to Builder instances using MethodType
    setattr(Builder, "build", build_method)

    return Builder  # type: ignore[return-value]


def all_subclasses(cls):
    """Get all subclasses of cls recursively."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


class BaseExperiment(BaseModel):
    """Base experiment class which require an infra and a 'run' method."""

    infra: exca.TaskInfra = exca.TaskInfra()

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return []

    def run(self):
        raise NotImplementedError


def run_grid(
    exp_cls: tp.Type[BaseExperiment],
    exp_name: str,
    base_config: dict[str, tp.Any],
    grid: dict[str, list],
    n_randomly_sampled: int | None = None,
    job_name_keys: list[str] | None = None,
    combinatorial: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    debug: bool = False,
    infra_mode: str = "retry",
    random_state: int | None = None,
) -> list[exca.ConfDict]:
    """Run grid over provided experiment.

    Parameters
    ----------
    exp_cls :
        Experiment class to instantiate with `grid`. Must have an `infra` attribute, which will be
        updated when instantiating the different experiments of the grid.
    exp_name :
        Name of the base experiment to run.
    grid :
        Dictionary containing values to perform the sweep on.
    n_randomly_sampled :
        If provided, number of randomly sampled configurations from the grid. If None, run full
        grid. See `random_state` parameter to seed the sampling.
    base_config :
        Base configuration to update.
    job_name_keys :
       Flattened config key(s) to update with the experiment-specific 'job_name' variable. E.g.,
       can be used to pass the job name to a wandb logger.
    combinatorial :
        If True, run grid over all possible combinations of the grid. If False, run each parameter
        change individually.
    overwrite :
        If True, delete existing experiment-specific folder.
    dry_run :
        If True, do not add tasks to the infra.
    debug :
        If True, bypass the infra.cluster and run the first experiment only locally. This is useful
        for quick sanity checking of the experiment configuration.
    infra_mode :
        Whether to rerun existing or failed experiments.
        - cached: cache is returned if available (error or not),
                otherwise computed (and cached)
        - retry: cache is returned if available except if it's an error,
                otherwise (re)computed (and cached)
        - force: cache is ignored, and result is (re)computed (and cached)
    random_state :
        Random state for random sampling of the grid.

    Returns
    -------
    list :
        List of config dictionaries used for each experiment of the grid.
    """
    job_array_kwargs = {}
    if dry_run or debug:
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("exca")) < Version("0.4.5"):
            raise ImportError("`dry_run` requires `exca>=0.4.5` to be installed.")
        job_array_kwargs["allow_empty"] = True

    if random_state is not None and n_randomly_sampled is None:
        warn(
            "`random_state` is provided but `n_randomly_sampled` is None. "
            "`random_state` will be ignored.",
        )

    # Update savedir of experiment infra
    base_config["infra"]["job_name"] = exp_name
    base_folder = Path(base_config["infra"]["folder"])
    if not all(isinstance(v, list) for v in grid.values()):
        raise ValueError("Grid values must be lists.")

    task: BaseExperiment = exp_cls(
        **base_config,
    )

    if n_randomly_sampled is not None:
        if n_randomly_sampled > prod(len(v) for v in grid.values()):
            raise ValueError("n_randomly_sampled is larger than the grid size.")
        rng = np.random.RandomState(random_state)
        grid_product = [
            {k: rng.choice(v) for k, v in grid.items()} for _ in range(n_randomly_sampled)
        ]

    else:
        if combinatorial:
            grid_product = list(
                dict(zip(grid.keys(), v)) for v in product(*grid.values())
            )
        else:
            grid_product = [
                {param: value} for param, values in grid.items() for value in values
            ]

    print(f"Launching {len(grid_product)} tasks")

    out_configs = []
    tmp = task.infra.clone_obj(**{"infra.mode": infra_mode})
    with tmp.infra.job_array(**job_array_kwargs) as tasks:
        for params in grid_product:
            config = exca.ConfDict(base_config)
            config.update(params)
            uid_suffix = config.to_uid()[-8:]
            job_name = exca.ConfDict(params).to_uid()[:-8] + uid_suffix

            folder = base_folder / exp_name / job_name
            if folder.exists():  # FIXME: adapt to checkpointing
                print(f"{folder} already exists.")
                if overwrite and not dry_run:
                    print(f"Deleting {folder}.")
                    shutil.rmtree(folder)
                    folder.mkdir()

            # Update infra and logger
            config["infra.folder"] = str(folder)
            if job_name_keys is not None:
                for key in job_name_keys:
                    config.update({key: str(job_name)})

            if not dry_run:
                task_ = exp_cls(**config)
                if debug:
                    task_.run()
                    out_configs.append(config)
                    break
                tasks.append(task_)

            out_configs.append(config)

    print("Done.")

    return out_configs


class CsvLoggerConfig(BaseModel):
    """
    Pydantic configuration for torch-lightning's CSVLogger.
    """

    name: str | None = "lightning_logs"
    version: int | str | None = None
    prefix: str = ""
    flush_logs_every_n_steps: int = 100

    def build(self, save_dir: str | Path):
        from lightning.pytorch.loggers import CSVLogger

        config = self.model_dump()
        return CSVLogger(**config, save_dir=save_dir)


class WandbLoggerConfig(BaseModel):
    """
    Pydantic configuration for torch-lightning's wandb logger.
    See https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html.
    If you want to resume a run, you can use the `id` field to specify the run id, either in the config or in the `build` method.
    """

    # core fields
    name: str | None = None
    group: str
    entity: str | None = None
    project: str | None = None
    # extra fields
    offline: bool = False
    host: str | None = None
    id: str | None = None
    dir: Path | None = None
    anonymous: bool | None = None
    log_model: str | bool = False
    experiment: tp.Any | None = None
    prefix: str = ""
    resume: tp.Literal["allow", "never", "must"] = "allow"

    # pylint: disable=redefined-builtin
    def build(
        self,
        save_dir: str | Path,
        xp_config: dict | pydantic.BaseModel | None = None,
        run_id: str | None = None,
    ) -> tp.Any:
        import wandb

        if self.offline:
            login_kwargs = {"key": "X" * 40}
        else:
            login_kwargs = {"host": self.host}  # type: ignore
        wandb.login(**login_kwargs)  # type: ignore
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(xp_config, pydantic.BaseModel):
            xp_config = xp_config.model_dump()
        config = self.model_dump()
        if run_id is not None:
            config["id"] = run_id
        del config["host"]
        logger = WandbLogger(**config, save_dir=save_dir, config=xp_config)
        try:
            logger.experiment.config["_dummy"] = None  # To launch initialization
        except TypeError:
            pass  # Crashes if called in a second process, e.g. with DDP
        return logger


class WandbInfra(exca.TaskInfra):
    wandb_config: WandbLoggerConfig | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.wandb_config and self.wandb_config.group is not None:
            # pylint: disable=attribute-defined-outside-init
            self.version = self.wandb_config.group

    def _wandb_uid(self):
        if self.wandb_config.group is not None and self.wandb_config.name is not None:
            uid = self.wandb_config.group + "-" + self.wandb_config.name
        else:
            uid = self.uid().split("-")[-1]
        for bad_char in "/:,=[]{}()":
            uid = uid.replace(bad_char, ".")
        return uid

    def _run_method(self, *args, **kwargs):
        out = super()._run_method(*args, **kwargs)

        if self.wandb_config is not None:
            import wandb

            try:
                with wandb.init(
                    project=self.wandb_config.project,
                    entity=self.wandb_config.entity,
                    group=self.wandb_config.group,
                    name=self.wandb_config.name,
                ) as run:
                    artifact = wandb.Artifact(self._wandb_uid(), type="pkl")
                    wandb_folder = self.uid_folder() / "wandb"
                    wandb_folder.mkdir(exist_ok=True, parents=True)
                    with open(wandb_folder / "output.pkl", "wb") as f:
                        pickle.dump(out, f)
                    self.config(uid=False, exclude_defaults=False).to_yaml(
                        wandb_folder / "config.yaml"
                    )
                    fnames = [wandb_folder / "config.yaml", wandb_folder / "output.pkl"]
                    try:
                        env = submitit.JobEnvironment()
                        fnames += [env.paths.stderr, env.paths.stdout]
                    except:
                        pass  # Not running in submitit
                    for fname in fnames:
                        artifact.add_file(fname)
                    run.log_artifact(artifact)
                    print(f"Uploaded to wandb: {self._wandb_uid()}")
                    (wandb_folder / "output.pkl").unlink()
            except wandb.errors.CommError:
                print("Could not connect to wandb. Skipping upload")
        return out

    def download(self, version="v0") -> tp.Any:
        if self.uid_folder().exists():  # type: ignore
            print(f"Folder {self.uid_folder()} already exists.")
            return
        if self.wandb_config is None:
            raise ValueError(
                "wandb_config must be provided to download artifacts from wandb."
            )
        import wandb

        with wandb.init(
            project=self.wandb_config.project, entity=self.wandb_config.entity
        ) as run:
            artifact = run.use_artifact(f"{self._wandb_uid()}:{version}")
            artifact.download(self.uid_folder())
        return artifact


def _is_constant_feature(
    var: torch.Tensor, mean: torch.Tensor, n_samples: torch.Tensor
) -> torch.Tensor:
    """Detect if a extractor is indistinguishable from a constant extractor (on torch Tensors).

    See `sklearn.preprocessing._data._is_constant_feature`.
    """
    eps = torch.finfo(torch.float32).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound


class StandardScaler(BaseModel):
    """Standard scaler that can be fitted by batch and handles 2-dimensional extractors."""

    dim: int = 1  # Dimension across which the statistics should be computed

    # Internal
    _mean: torch.Tensor | None = None
    _var: torch.Tensor | None = None
    _scale: torch.Tensor | None = None
    _original_shape: list | None = None
    _n_samples_seen: int = 0

    def _reset(self):
        self._mean = None
        self._var = None
        self._scale = None
        self._original_shape = None
        self._n_samples_seen = 0

    def _transpose_flatten(self, X: torch.Tensor) -> torch.Tensor:
        """Transpose and flatten to have (n_total_examples, n_latent_dims)."""
        if X.ndim > 2:
            self._original_shape = [s for i, s in enumerate(X.shape) if i != self.dim]
            X = X.transpose(self.dim, -1).flatten(end_dim=-2)
        return X

    def _unflatten_untranspose(self, X: torch.Tensor) -> torch.Tensor:
        if self._original_shape is not None:
            X = X.unflatten(dim=0, sizes=self._original_shape).transpose(self.dim, -1)
        return X

    def partial_fit(self, X: torch.Tensor) -> StandardScaler:
        X = self._transpose_flatten(X)
        m = self._n_samples_seen
        n = X.shape[0]

        # Update mean
        previous_mean = (
            torch.zeros(X.shape[1], device=X.device) if self._mean is None else self._mean
        )
        batch_mean = X.mean(dim=0)
        self._mean = (m / (m + n)) * previous_mean + (n / (m + n)) * batch_mean

        # Update variance
        previous_var = (
            torch.zeros(X.shape[1], device=X.device) if self._var is None else self._var
        )
        self._var = (
            (m / (m + n)) * previous_var
            + (n / (m + n)) * X.var(dim=0)
            + (m * n / (m + n) ** 2) * (previous_mean - batch_mean) ** 2
        )
        scale = self._var.sqrt()  # type: ignore

        # Compute near-constant mask to avoid scaling by 0
        constant_mask = _is_constant_feature(self._var, self._mean, self._n_samples_seen)  # type: ignore
        scale[constant_mask] = 1.0
        self._scale = scale  # type: ignore
        self._n_samples_seen += n

        return self

    def fit(self, X: torch.Tensor) -> StandardScaler:
        self._reset()
        return self.partial_fit(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clone()
        X = self._transpose_flatten(X)
        X = (X - self._mean.to(X.device)) / self._scale.to(X.device)  # type: ignore
        X = self._unflatten_untranspose(X)
        return X


X = tp.TypeVar("X")


class TimedIterator(tp.Generic[X]):
    """Keeps last fetch durations of the iterator, as well as last call to call
    durations.
    This is handy to investigate ratio spent in a dataloader compared to the whole
    training loop.

    Parameters
    ----------
    iterable: iterable
        The iterable to analyze, usually a torch Dataloader
    store_last: int
        maximum number of durations to keep in memory

    Note
    ----
    estimated_ratio is based on mean values, you may want check
    - last_calls: last durations of the iterable call
    - last_loops: last durations of the full loop back to the iterable
    """

    def __init__(self, iterable: tp.Iterable[X], store_last: int = 100) -> None:
        self._iterable = iterable
        self._iterator = iter(self._iterable)
        self.last_calls: collections.deque[float] = collections.deque(maxlen=store_last)
        self.last_loops: collections.deque[float] = collections.deque(maxlen=store_last)
        self._last_call_time: float | None = None

    def __next__(self) -> X:
        t0 = time.time()
        x = next(self._iterator)
        self.last_calls.append(time.time() - t0)
        if self._last_call_time is not None:
            self.last_loops.append(t0 - self._last_call_time)
        self._last_call_time = t0
        return x

    def __len__(self) -> int:
        return len(self._iterable)  # type: ignore

    def estimated_ratio(self) -> float:
        return float(np.mean(self.last_calls) / np.mean(self.last_loops))

    def __iter__(self) -> tp.Iterator[X]:
        self._iterator = iter(self._iterable)
        self.last_calls.clear()
        self.last_loops.clear()
        return self
