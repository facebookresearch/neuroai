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

Extra dependency groups are available for dataset downloading and
model loading:

```bash
pip install 'neuralbench-repo/.[datasets,models]'
```

## First-run configuration

The first time you run `neuralbench`, you will be prompted to set three
paths:

- **`DATA_DIR`** -- where datasets are downloaded.
- **`CACHE_DIR`** -- where preprocessed data is cached.
- **`SAVE_DIR`** -- where results are saved.

The configuration is stored in `~/.neuralbench/config.json` by default.

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
