(neuralset-install)=
# Installation

## Basic Install

```bash
pip install neuralset neuralfetch
```

**neuralset** is the data pipeline. **neuralfetch** adds the dataset
catalog — you need both to follow the tutorials.

Verify it works:

```python
import neuralset as ns
print(ns.Study.catalog())
```

To follow the tutorials (which use spacy embeddings and audio data):

```bash
pip install 'neuralset[tutorials]'
```

Some extractors require heavier dependencies. Install everything at once
with:

```bash
pip install 'neuralset[all]'
```

Key extras in `[all]` beyond `[tutorials]`: `transformers` (HuggingFace
text/image/audio models), `torchaudio`, `torchvision`, `pybids` (BIDS
fMRI datasets).

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
uv pip install pip   # spacy auto-download requires pip inside the venv
```

**Step 2:** Install packages in editable mode:

```bash
# from the repo root — core pipeline + datasets
uv pip install -e 'neuralset-repo/.[dev,all]'
uv pip install -e 'neuralfetch-repo/.'
```

**Step 3:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuralset-repo/.[dev]'
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
conda create -n neuralset python=3.12
conda activate neuralset
```

**Step 2:** Install pip and uv in the environment:

```bash
conda install pip
pip install uv
```

**Step 3:** Install packages in editable mode:

```bash
# from the repo root — core pipeline + datasets
uv pip install -e 'neuralset-repo/.[dev,all]'
uv pip install -e 'neuralfetch-repo/.'
```

**Step 4:** For strict type checking (optional):

```bash
uv pip install --config-settings editable_mode=strict -e 'neuralset-repo/.[dev]'
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


```{raw} html
<div class="page-nav">
  <a href="index.html" class="page-nav-btn page-nav-btn--outline">&larr; Overview</a>
  <a href="auto_examples/walkthrough/index.html" class="page-nav-btn">Start Tutorials &rarr;</a>
</div>
```
