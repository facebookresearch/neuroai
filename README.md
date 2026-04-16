# neuroai

[![CI](https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/neuroai/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-facebookresearch.github.io%2Fneuroai-blue)](https://facebookresearch.github.io/neuroai/)

**From raw neuroimaging data to state-of-the-art encoding and decoding.** A Python suite for modern neuroscience research.

## 📦 Install

```bash
pip install neuralset          # core pipeline
pip install neuralfetch        # public dataset access
pip install neuraltrain        # deep learning models
```

## 📦 Core Packages

### 🧩 **neuralset** — Turn raw neuroimaging data into AI-ready datasets
- **Data loading** from local and remote sources (BIDS, custom formats)
- **Event-driven processing** with rich DataFrame semantics
- **Multi-modal extractors** for MEG, EEG, fMRI, EMG, iEEG, text, audio, video
- **Composable transformations** for alignment, filtering, and enrichment
- **Torch-native segmentation** for efficient batching

### ⚡ **neuraltrain** — Deep learning for neuroimaging
- **PyTorch + Lightning** for scalable model training
- **Specialized architectures** for MEG, EEG, fMRI, and multi-modal data
- **Custom augmentations** for realistic data transformations
- **Domain-specific loss functions** and metrics
- **Multi-GPU training** with distributed backends

### 🌐 **neuralfetch** — Access public brain datasets
- **Interfaces to public datasets** (OpenNeuro, DANDI, OSF, MOABB, ...)
- **One-command downloads** with automated caching and verification
- **Dataset versioning** for reproducible research
- **Metadata enrichment** and study introspection

## 📚 Documentation
[**Documentation**](https://facebookresearch.github.io/neuroai/)

Infrastructure: [exca documentation](https://facebookresearch.github.io/exca/)

## 🛠️ Development

Pre-commit hooks:
```bash
pre-commit install
```

Linting and testing:
```bash
ruff check neuralset   # Lint code
ruff format neuralset  # Format code
mypy neuralset         # Type checking
pytest neuralset       # Run tests
```

## 📂 Project Structure
- **neuralset-repo/** — Data loading, transforms, extractors
- **neuralfetch-repo/** — Dataset interfaces and downloads
- **neuraltrain-repo/** — Deep learning models and training
- **docs/** — Sphinx documentation

## Third-Party Content
References to third-party content from other locations are subject to
their own licenses and you may have other legal obligations or
restrictions that govern your use of that content.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
