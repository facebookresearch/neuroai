# neuralset

A Python toolkit for building neuroimaging data pipelines — from raw data
to AI-ready batched tensors.

- **Event-driven processing** with typed, validated DataFrames
- **Multi-modal extractors** for MEG, EEG, fMRI, EMG, iEEG, text, image, audio, video
- **Caching and remote compute** via [exca](https://github.com/facebookresearch/exca)

## Install

```bash
pip install neuralset
```

## Quick example

```python
import neuralset as ns

# Load a study and build events
study = ns.Study(name="Mne2013Sample", path="./data")

study.download() # ~ 1.6GB

events = study.run()

# Configure extractors
meg = ns.extractors.MegExtractor(frequency=100.0, filter=(0.5, 30))

# Segment around triggers and build a dataset
segmenter = ns.Segmenter(
    extractors=dict(meg=meg),
    trigger_query='type == "Word"',
    start=-0.1, duration=0.5,
)
dataset = segmenter.apply(events)
batch = dataset.load_all()
```

## Citation

Citation coming soon.

## Third-Party Content

References to third-party content from other locations are subject to
their own licenses and you may have other legal obligations or
restrictions that govern your use of that content.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
