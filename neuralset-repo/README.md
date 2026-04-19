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

If you use `NeuralSet` in your research, please cite [NeuralSet: A High-Performing Python Package for Neuro-AI](https://kingjr.github.io/files/neuralset.pdf):


```bibtex
@article{king2025neuralset,
  title   = {NeuralSet: A High-Performing Python Package for Neuro-AI},
  author  = {King, J-R. and Bel, C. and Evanson, L. and Gadonneix, J. and Houhamdi, S. and L{\'e}vy, J. and Raugel, J. and Santos Revilla, A. and Zhang, M. and Bonnaire, J. and Caucheteux, C. and D{\'e}fossez, A. and Desbordes, T. and Diego-Sim{\'o}n, P. and Khanna, S. and Millet, J. and Orhan, P. and Panchavati, S. and Ratouchniak, A. and Thual, A. and Brooks, T. and Begany, K. and Benchetrit, Y. and Careil, M. and Banville, H. and d'Ascoli, S. and Dahan, S. and Rapin, J.},
  year    = {2025},
  url     = {https://kingjr.github.io/files/neuralset.pdf}
}
```


## Third-Party Content

References to third-party content from other locations are subject to
their own licenses and you may have other legal obligations or
restrictions that govern your use of that content.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
