# Project Example

`neuraltrain-repo/project_example/` is the reference template for a small
training project built on top of `neuralset`, `neuraltrain`, and
PyTorch Lightning.

It uses the [MNE sample dataset](https://mne.tools/stable/documentation/datasets.html#sample)
to train a 4-class MEG stimulus classifier, and it shows the expected
structure of a real experiment package.

## What It Includes

| File | Role |
|------|------|
| `main.py` | `Data` and `Experiment` pydantic models |
| `pl_module.py` | `BrainModule` Lightning wrapper |
| `grids/defaults.py` | default config dict for a single local run |
| `grids/run_grid.py` | hyperparameter sweep via `run_grid` |
| `plot_figure.py` | sketch for loading and plotting saved outputs |

## Data

`Data` is a small pydantic model that holds a **study chain** and a
**`Segmenter`**, then builds DataLoaders for each split:

```python
class Data(pydantic.BaseModel):
    study: ns.Step
    segmenter: ns.dataloader.Segmenter
    batch_size: int = 64
    num_workers: int = 0

    def build(self) -> dict[str, DataLoader]:
        events = self.study.run()
        dataset = self.segmenter.apply(events)
        dataset.prepare()

        loaders = {}
        for split, shuffle in [("train", True), ("val", False), ("test", False)]:
            ds = dataset.select(dataset.triggers["split"] == split)
            loaders[split] = DataLoader(
                ds, collate_fn=ds.collate_fn,
                batch_size=self.batch_size, shuffle=shuffle,
                num_workers=self.num_workers,
            )
        return loaders
```

The study chain in the default config is `Mne2013Sample` piped into
`SklearnSplit`, which adds a `split` column to the events table.
The `Segmenter` holds the extractors (`MegExtractor` for input,
`EventField` for target) and the segment window.

## Experiment

`Experiment` is a plain `pydantic.BaseModel` that collects all training
pieces and orchestrates the run:

```python
class Experiment(pydantic.BaseModel):
    data: Data
    brain_model_config: BaseModelConfig
    loss: BaseLoss
    optim: LightningOptimizer
    metrics: list[BaseMetric]
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def run(self) -> dict[str, float | None]:
        loaders = self.data.build()
        brain_module = self._build_brain_module(loaders["train"])
        trainer = self._setup_trainer()
        if not self.test_only:
            self.fit(brain_module, trainer, loaders["train"], loaders["val"])
        return self.test(brain_module, trainer, loaders["test"])
```

Key design choices:

- **Stateless**: `brain_module` and `trainer` are local variables passed
  between methods, not stored as private attributes.
- **`TaskInfra`** (from `exca`): controls folder, cluster, GPU count, and
  caching. `@infra.apply` makes `run()` cacheable and Slurm-submittable.
- **Configurable logging**: `csv_config` and `wandb_config` are optional;
  when both are `None`, a `DummyLogger` is used.
- **Returns results**: `run()` returns a `dict[str, float | None]` of test
  metrics instead of the trainer object.

## Run a Local Example

From the repository root:

```bash
cd neuraltrain-repo
python -m project_example.grids.defaults
```

This runs a local experiment with the default configuration.

## Run a Grid

```bash
cd neuraltrain-repo
python -m project_example.grids.run_grid
```

This launches the example sweep and shows how `neuraltrain.utils.run_grid()`
is used to expand experiment configurations over depth, epoch count, and seed.

## Why This Example Matters

The example is the clearest starting point if you want to:

- wire a `neuralset` study chain into training via `Segmenter`
- keep models, metrics, and optimizers as typed config objects
- add CSV or W&B logging through config fields
- launch Slurm jobs or grid sweeps through `exca.TaskInfra`
- organize experiments as reusable, stateless pydantic classes

```{raw} html
<div class="page-nav">
  <a href="auto_examples/trainer/04_trainer.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorials: Trainer</a>
</div>
```
