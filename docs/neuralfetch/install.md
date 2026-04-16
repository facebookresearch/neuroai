(neuralfetch-install)=
# Installation

## Basic Install

```bash
pip install neuralfetch
```

Then verify it works:

```python
import neuralset as ns

# List available studies
all_studies = ns.Study.catalog()
print(f"{len(all_studies)} studies available")
```

To run the example tutorials, also install `neuralset` with the `tutorials` extra:

```bash
pip install 'neuralset[tutorials]'
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

**Step 2:** Install neuralfetch in editable mode:

```bash
# from the repo root
uv pip install -e 'neuralfetch-repo/.'
```

**Step 3:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuralfetch-repo/.'
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
conda create -n neuralfetch python=3.12
conda activate neuralfetch
```

**Step 2:** Install pip and uv in the environment:

```bash
conda install pip
pip install uv
```

**Step 3:** Install neuralfetch in editable mode:

```bash
# from the repo root
uv pip install -e 'neuralfetch-repo/.'
```

**Step 4:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuralfetch-repo/.'
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

      // Deactivate all buttons and tabs
      document.querySelectorAll('.install-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.install-tab').forEach(t => t.classList.remove('active'));

      // Activate selected
      e.target.classList.add('active');
      document.getElementById(`tab-${tab}`).classList.add('active');
    });
  });
</script>

```


```{raw} html
<div class="page-nav">
  <a href="index.html" class="page-nav-btn page-nav-btn--outline">&larr; Overview</a>
  <a href="auto_examples/index.html" class="page-nav-btn">Tutorials &rarr;</a>
</div>
```
