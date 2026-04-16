(neuraltrain-install)=
# Installation

## Basic Install

```bash
pip install neuraltrain
```

Then verify it works:

```python
import neuraltrain
from neuraltrain import models

model_cfg = models.SimpleConvTimeAgg(hidden=32, depth=4, merger_config=None)
print(model_cfg)
```

## Developer Install

For active development or to try the latest features:

```{raw} html

<div class="install-tabs">
  <div class="install-tabs__buttons">
    <button class="install-tab-btn active" data-tab="uv">📦 uv (fast)</button>
    <button class="install-tab-btn" data-tab="conda">🐍 conda</button>
  </div>

  <div id="tab-uv" class="install-tab active">

```

**Step 1:** Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

**Step 2:** Install neuraltrain in editable mode:

```bash
# from the repo root
uv pip install -e 'neuraltrain-repo/.[dev]'

# recommended: dev + all extras (models, metrics, lightning) — needed for all tutorials
uv pip install -e 'neuraltrain-repo/.[dev,all]'
```

**Step 3:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuraltrain-repo/.[dev,all]'
```

**Step 4:** Set up pre-commit hooks:

```bash
pre-commit install
```

```{raw} html
  </div>

  <div id="tab-conda" class="install-tab">

```

**Step 1:** Create and activate a conda environment:

```bash
conda create -n neuraltrain python=3.12
conda activate neuraltrain
```

**Step 2:** Install pip and uv in the environment:

```bash
conda install pip
pip install uv
```

**Step 3:** Install neuraltrain in editable mode:

```bash
# from the repo root
uv pip install -e 'neuraltrain-repo/.[dev]'

# recommended: dev + all extras (models, metrics, lightning) — needed for all tutorials
uv pip install -e 'neuraltrain-repo/.[dev,all]'
```

**Step 4:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuraltrain-repo/.[dev,all]'
```

**Step 5:** Set up pre-commit hooks:

```bash
pre-commit install
```

```{raw} html
  </div>
</div>

<script>
  document.querySelectorAll('.install-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const tab = e.target.dataset.tab;
      document.querySelectorAll('.install-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.install-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      document.getElementById(`tab-${tab}`).classList.add('active');
    });
  });
</script>

```

## Optional Extras

- `lightning` — PyTorch Lightning and Weights & Biases support.
- `models` — optional dependencies for some model families (e.g. `braindecode`, `x_transformers`, `green`).
- `dev` — testing and documentation tools.

## Notes

- If a specific optional model fails to import, check that the corresponding
  extra (`.[models]`) is installed in the current environment.
- The project example requires the `lightning` extra to run.

```{raw} html
<div class="page-nav">
  <a href="index.html" class="page-nav-btn page-nav-btn--outline">&larr; Overview</a>
  <a href="tutorials.html" class="page-nav-btn">Tutorials &rarr;</a>
</div>
```
