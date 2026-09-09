# Installation

## Prerequisites

- Python >= 3.12

## Install from PyPI

```bash
pip install neuralbench
```

## Install from source

From the monorepo root:

```bash
pip install ./neuralbench-repo
```

Or from inside the sub-repo:

```bash
cd neuralbench-repo
pip install .
```

(Use `pip install -e .` instead if you intend to modify the source -- see
[Developer install](#developer-install) below.)

## Developer install

Editable mode picks up local source changes without reinstalling, and the
`[dev]` extra brings in `pytest`, `ruff`, `mypy`, `pre-commit`, and the
type stubs that `mypy neuralbench` requires:

```bash
pip install -e 'neuralbench-repo/.[dev]'
pre-commit install
```

## Optional dependencies

The base install is enough to download datasets (via `neuralfetch[quickstart]`),
load pretrained model weights (via `braindecode[hub]`), and run and score every
task. The only extra a user may want is `wandb`, for the optional experiment
tracking described below:

```bash
pip install 'neuralbench-repo/.[wandb]'
```

## First-run configuration

The first time you run `neuralbench`, you will be prompted to set three
paths:

- **`DATA_DIR`** -- where datasets are downloaded.
- **`CACHE_DIR`** -- where preprocessed data is cached.
- **`SAVE_DIR`** -- where results are saved.

The configuration is stored in `~/.neuralbench/config.json` by default.

### Execution backend (SLURM vs. local)

`neuralbench` dispatches preparation and training jobs through the `CLUSTER`
key in `~/.neuralbench/config.json`:

- **`"auto"`** (default) -- submit to SLURM when it is auto-detected, otherwise
  run locally. Non-debug SLURM runs additionally require `SLURM_PARTITION` to be
  set in the config.
- **`null`** -- force everything (training plus the preprocessing/target caches)
  to run locally, in-process, even on a SLURM cluster. Unlike `--debug`, this
  keeps the full config (full epochs and batches). `--prepare` likewise builds
  caches locally when `CLUSTER` is `null`.
- **`"slurm"`** -- always submit to SLURM.

For example, to run the full benchmark locally without SLURM, set:

```json
{
  "CLUSTER": null
}
```

### Weights & Biases (optional)

W&B logging is off unless `WANDB_HOST` is set in your config, and nothing
requires it: results are written to `SAVE_DIR` either way and stay accessible
through `--plot-cached`. To opt in, install the `wandb` extra and set
`WANDB_HOST` to your host.

### Custom config location

Set the `NEURALBENCH_CONFIG` environment variable to point at a different
file (useful on shared machines, for CI, or when juggling multiple
profiles):

```bash
export NEURALBENCH_CONFIG=/path/to/my/neuralbench-config.json
neuralbench eeg audiovisual_stimulus --debug
```

The variable is read every time `neuralbench` starts, so you can switch
configs by re-exporting it.
