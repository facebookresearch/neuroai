<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuroai_anim_dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/neuroai_anim_light.gif">
    <img alt="neuroai" src="docs/_static/neuroai_anim_light.gif" width="280">
  </picture>
</p>

<h3 align="center">The Python suite for brain-AI research</h3>
<p align="center"><sub>Simple &nbsp;·&nbsp; Fast &nbsp;·&nbsp; Robust &nbsp;·&nbsp; Scalable</sub></p>

<p align="center">
  <a href="https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml"><img src="https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://facebookresearch.github.io/neuroai/"><img src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/facebookresearch/neuroai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+"></a>
</p>

<p align="center">
  <a href="#packages">Packages</a> &nbsp;·&nbsp;
  <a href="#related-projects">Related projects</a> &nbsp;·&nbsp;
  <a href="https://facebookresearch.github.io/neuroai/">Documentation</a>
</p>

---

neuroai is a modular Python suite for brain-AI research. It covers the full pipeline: accessing curated public brain datasets, building typed & cacheable feature pipelines across all recording modalities (MEG, EEG, fMRI, iEEG, EMG) and stimulus types (text, images, audio, video), and training deep-learning models — with a single unified interface.

<br>

<p align="center">
  <a href="https://facebookresearch.github.io/neuroai/">
    <img src="https://img.shields.io/badge/📖%20Explore%20the%20full%20documentation-facebookresearch.github.io%2Fneuroai-448aff?style=for-the-badge" alt="Explore the full documentation">
  </a>
</p>
<p align="center">
  <sub>Interactive quickstarts &nbsp;·&nbsp; Step-by-step tutorials &nbsp;·&nbsp; Complete API reference<br>
  Pick a task, a modality, a dataset — the docs generate the code for you.</sub>
</p>

<br>

<p align="center">
  <img src="docs/_static/neuralset.gif" alt="neuralset demo" width="720">
</p>

---

## Packages

Each pipeline step maps to a dedicated package:

<table width="100%">
<tr>
<td align="center" valign="top" width="33%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuralfetch_dark.png">
  <img src="docs/_static/neuralfetch_light.png" width="120" alt="neuralfetch">
</picture>
<br><br>
<strong><a href="https://facebookresearch.github.io/neuroai/neuralfetch/index.html">neuralfetch</a></strong><br><br>
<sub>Access the world's curated brain datasets.<br>
19+ studies from OpenNeuro, DANDI, OSF,<br>
HuggingFace, Zenodo and more.</sub>
<br><br>

```bash
pip install neuralfetch
```

<br>
</td>
<td align="center" valign="top" width="33%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuralset_dark.png">
  <img src="docs/_static/neuralset_light.png" width="120" alt="neuralset">
</picture>
<br><br>
<strong><a href="https://facebookresearch.github.io/neuroai/neuralset/index.html">neuralset</a></strong><br><br>
<sub>Turn brain data into AI-ready features.<br>
Events, extractors, transforms &amp;<br>
segmentation into PyTorch datasets.</sub>
<br><br>

```bash
pip install neuralset
```

<br>
</td>
<td align="center" valign="top" width="33%">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/neuraltrain_dark.png">
  <img src="docs/_static/neuraltrain_light.png" width="120" alt="neuraltrain">
</picture>
<br><br>
<strong><a href="https://facebookresearch.github.io/neuroai/neuraltrain/index.html">neuraltrain</a></strong><br><br>
<sub>Deep learning for the brain, supercharged.<br>
ConvNets, Transformers, losses, metrics<br>
&amp; multi-GPU training (PyTorch + Lightning).</sub>
<br><br>

```bash
pip install neuraltrain
```

<br>
</td>
</tr>
</table>

---

## Related projects

- **[exca](https://facebookresearch.github.io/exca/)** — Execution & caching framework powering neuroai's backbone

---

## License

This project is licensed under the [MIT License](LICENSE).

<sub>References to third-party content are subject to their own licenses.</sub>

---

<p align="center">
  <sub>Built with ❤️ at <a href="https://ai.meta.com/">Meta AI</a></sub>
</p>
