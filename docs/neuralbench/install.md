# Installation

## Prerequisites

- Python >= 3.12
- For GPU training, an NVIDIA driver new enough for the `torch` wheel that pip
  resolves. `pip install neuralbench` takes the default PyPI `torch`, which
  tracks the newest CUDA release, so an older driver fails on every GPU call.
  Check with `python -c "import torch; print(torch.cuda.get_device_capability(0))"`
  -- if it raises (typically `The NVIDIA driver on your system is too old`),
  install a build matching your driver, for example CUDA 12.6:

  ```bash
  pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
  ```

  Every training run needs a working GPU, `--debug` included: there is no CPU
  fallback, so a mismatched driver stops the quick sanity check each task page
  opens with. `--download` and `--plot-cached` need none of this, and neither
  does `--prepare` for most tasks -- but one whose target extractor runs a
  vision or audio model, as `eeg image` embeds its stimuli with DINOv2, uses
  the GPU to build that cache.

## Install from PyPI

```bash
pip install neuralbench
```

## Install with uv

[uv](https://docs.astral.sh/uv/) is a drop-in alternative to `pip` and every
route on this page has a `uv` equivalent, with no `neuralbench`-side
configuration. It also fetches its own CPython, which is the easy way out if
the system Python is older than 3.12:

```bash
# 1. Get uv (or `pip install uv`, if your network blocks astral.sh)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate a Python 3.12 environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install, with the optional extras from below
uv pip install 'neuralbench[wandb]' 'moabb>=1.7.1' 'eegdash>=0.8.2'

# ... or from a monorepo checkout, editable for development
uv pip install ./neuralbench-repo
uv pip install -e 'neuralbench-repo/.[dev]'

# 4. Check it landed
python -c "import importlib.metadata as m; print(m.version('neuralbench'))"
neuralbench --help
```

For the driver-matched `torch` described above, use uv's `--torch-backend`
rather than the `--index-url` from the `pip` recipe:

```bash
uv pip install --torch-backend=cu126 --reinstall-package torch \
  --reinstall-package torchvision --reinstall-package torchaudio \
  torch torchvision torchaudio
```

Both give you the same `+cu126` build, but `--index-url` *replaces* PyPI in uv
instead of adding to it, so every transitive dependency gets resolved from the
PyTorch mirror too -- which quietly downgrades `numpy`, `setuptools` and
`filelock`. `--torch-backend=auto` picks the build matching the detected
driver.

One difference to know about: uv does not read `pip.conf`. If your packages
come from an internal mirror configured there, pass `--extra-index-url` or set
`UV_EXTRA_INDEX_URL` explicitly.

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
[Developer install](developer-install) below.)

(developer-install)=
## Developer install

Editable mode picks up local source changes without reinstalling, and the
`[dev]` extra brings in `pytest`, `ruff`, `mypy`, `pre-commit`, and the
type stubs that `mypy neuralbench` requires:

```bash
pip install -e 'neuralbench-repo/.[dev]'
pre-commit install
```

## Optional dependencies

The base install loads pretrained model weights (via `braindecode[hub]`) and
downloads most datasets (via `neuralfetch[quickstart]`). Two dataset families
reach their host through a client `neuralbench` does not depend on:

| Package | Needed by |
| --- | --- |
| `moabb>=1.7.1` | every MOABB-backed EEG dataset, including the `eeg motor_imagery` default |
| `eegdash>=0.8.2` | every EEG-Dash-served dataset, including `emg pose` |

The error names the missing package, so installing on demand works; to have
both up front:

```bash
pip install 'moabb>=1.7.1' 'eegdash>=0.8.2'
```

`wandb` is the package's only runtime extra -- `dev` and `docs` exist for
working on `neuralbench` itself -- and it enables the optional experiment
tracking described below:

```bash
pip install 'neuralbench[wandb]'
```

## First-run configuration

The first time you run `neuralbench`, you will be prompted to set three
paths:

- **`DATA_DIR`** -- where datasets are downloaded.
- **`CACHE_DIR`** -- where preprocessed data is cached.
- **`SAVE_DIR`** -- where results are saved.

The configuration is stored in `~/.neuralbench/config.json` by default, and the
three directories are created if they do not exist.

A config file you write yourself has to define everything the prompt would have
written. These six keys have no default, and a missing one fails with a bare
`KeyError`:

```json
{
  "USER": "your-username",
  "ENTITY_NAME": "your-username",
  "PROJECT_NAME": "neuralbench",
  "DATA_DIR": "/path/to/data",
  "CACHE_DIR": "/path/to/cache",
  "SAVE_DIR": "/path/to/results"
}
```

`USER`, `ENTITY_NAME` and `PROJECT_NAME` only label runs and W&B entries, so
any string does. Every remaining key -- `CLUSTER`, `SLURM_PARTITION`,
`SLURM_CONSTRAINT`, `N_CPUS`, `WANDB_HOST` -- is optional.

The prompt needs a terminal. Where stdin is not one -- a SLURM batch script,
`nohup`, CI, some notebooks -- `neuralbench` skips it, prints a notice, and
falls back to `/tmp/neuralbench/{data,cache,save}`. Write the config file
beforehand, or point `NEURALBENCH_CONFIG` at one, to keep such runs off local
disk.

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

For example, to run the full benchmark locally without SLURM, add this
alongside the required keys above:

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
