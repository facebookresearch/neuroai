<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuroai_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/neuroai_light.png">
    <img alt="neuroai" src="docs/_static/neuroai_light.png" width="160">
  </picture>
</p>

<p align="center">
  <strong>The Python suite for brain data — from raw recordings to state-of-the-art decoding.</strong><br>
  <sub>Simple. Fast. Robust. Scalable.</sub>
</p>

<p align="center">
  <a href="https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml"><img src="https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://facebookresearch.github.io/neuroai/"><img src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/facebookresearch/neuroai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://facebookresearch.github.io/neuroai/">Documentation</a> •
  <a href="#install">Install</a> •
  <a href="#packages">Packages</a> •
  <a href="#development">Development</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is neuroai?

neuroai is a modular Python suite for neuroimaging research and AI. It handles the full pipeline: **downloading** public datasets, **loading and transforming** brain recordings (MEG, EEG, fMRI, iEEG, EMG) alongside stimuli (text, images, audio, video), and **training** deep-learning models — all with a unified, typed, cacheable interface.

Each pipeline step maps to a dedicated package:

<table>
<tr>
<td align="center" width="36%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuralfetch_dark.png">
  <img src="docs/_static/neuralfetch_light.png" width="120">
</picture>
<br><br>
<strong>neuralfetch</strong><br>
<sub>Download 19+ curated datasets from<br>OpenNeuro, OSF, HuggingFace…</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuralset_dark.png">
  <img src="docs/_static/neuralset_light.png" width="120">
</picture>
<br><br>
<strong>neuralset</strong><br>
<sub>Events, extractors, transforms &<br>segmentation into PyTorch datasets</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuraltrain_dark.png">
  <img src="docs/_static/neuraltrain_light.png" width="120">
</picture>
<br><br>
<strong>neuraltrain</strong><br>
<sub>ConvNets, Transformers, losses,<br>metrics & fast multi-GPU training</sub>
<br><br>
</td>
</tr>
</table>

> **📖 [Explore the full documentation →](https://facebookresearch.github.io/neuroai/)**
>
> Interactive quickstarts, step-by-step tutorials, and complete API reference.
> The docs generate ready-to-run code for any task × modality × dataset combination.

---

## Install

```bash
pip install neuralset neuralfetch neuraltrain
```

Optional extras for tutorials and all extractors:

```bash
pip install 'neuralset[tutorials]'   # spacy, matplotlib, soundfile
pip install 'neuralset[all]'         # all optional extractors
```

---

## Packages

| Package | What it does | Install |
|---------|-------------|---------|
| **[neuralset](https://facebookresearch.github.io/neuroai/neuralset/index.html)** | Core pipeline: events, extractors, transforms, segmentation, dataloaders | `pip install neuralset` |
| **[neuralfetch](https://facebookresearch.github.io/neuroai/neuralfetch/index.html)** | Public dataset catalog & download from 12+ sources | `pip install neuralfetch` |
| **[neuraltrain](https://facebookresearch.github.io/neuroai/neuraltrain/index.html)** | Deep-learning models, training loops, losses & metrics (PyTorch + Lightning) | `pip install neuraltrain` |

---

## Project structure

```
neuroai/
├── neuralset-repo/       # Core pipeline: events, extractors, transforms
├── neuralfetch-repo/     # Dataset catalog and download
├── neuraltrain-repo/     # Models, training loops, metrics
└── docs/                 # Sphinx documentation
```

---

## Development

```bash
git clone https://github.com/facebookresearch/neuroai.git
cd neuroai

# Create a venv (uv recommended)
uv venv .venv && source .venv/bin/activate
uv pip install pip                             # needed for spacy model downloads

# Install all packages in editable mode
uv pip install -e 'neuralset-repo/.[dev,all]'
uv pip install -e 'neuralfetch-repo/.'
uv pip install -e 'neuraltrain-repo/.[dev,all]'

# Verify
pre-commit install
pytest neuralset-repo/neuralset -x
```

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

```bash
ruff check .          # lint
ruff format .         # format
mypy neuralset-repo/  # type check
pytest -x             # test
```

---

## Related projects

- **[exca](https://facebookresearch.github.io/exca/)** — Execution & caching framework powering neuroai's compute graph
- **[MNE-Python](https://mne.tools/)** — Electrophysiology analysis (used internally for MEG/EEG I/O)

---

<details>
<summary><b>Demo</b></summary>
<p align="center">
  <img src="docs/_static/demo.gif" alt="neuroai code demo" width="700">
</p>
<p align="center">
  <img src="docs/_static/neuralset.gif" alt="neuralset terminal demo" width="700">
</p>
</details>

---

## License

This project is licensed under the [MIT License](LICENSE).

### Third-Party Content

References to third-party content from other locations are subject to
their own licenses and you may have other legal obligations or
restrictions that govern your use of that content.

---

<p align="center">
  <sub>Built with ❤️ at <a href="https://ai.meta.com/">Meta AI</a></sub>
</p>
