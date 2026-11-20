# neuralfetch

Download backends and data-fetching utilities for neuroimaging datasets.

- **Download backends** for 16 sources: DANDI, DataLad, Donders, Dryad, EEGDash, Figshare, GIN, Globus, Hugging Face, NEMAR, OpenNeuro, OSF, PhysioNet, S3, Synapse and Zenodo
- **Helper utilities** for study data management, BIDS export, and file scanning
- Built on top of [neuralset](https://github.com/facebookresearch/neuroai)

## Install

```bash
pip install neuralfetch
```

## Download permissions

Downloads made through `download_file`, including Zenodo files, keep Pooch's
private file mode by default. To opt into group-readable/writable completed
files on POSIX systems:

```bash
umask 002
NEURALFETCH_DOWNLOAD_FILE_MODE=0660 neuralfetch download MyStudy
```

The umask controls normally created directories and extracted zip members;
`NEURALFETCH_DOWNLOAD_FILE_MODE` sets exact permissions on newly transferred
files only after the transfer and any requested checksum verification succeed.
Use `0664` only when other users should also have read access. Cache hits keep
existing permissions; failed transfers and incomplete temporary files are
unchanged. This setting does not apply to
backends using other clients, such as S3 or NEMAR.

Python callers can override the environment with `download_file(...,
file_mode=0o660)` or `Zenodo(..., file_mode=0o660)`. Permission errors are raised
and do not produce a download success marker.

## Citation

```bibtex
@misc{king2026neuralset,
  title  = {NeuralSet: A High-Performing Python Package for Neuro-AI},
  author = {King, Jean-R{\'e}mi and Bel, Corentin and Evanson, Linnea
            and Gadonneix, Julien and Houhamdi, Sophia and L{\'e}vy, Jarod
            and Raugel, Josephine and Santos Revilla, Andrea
            and Zhang, Mingfang and Bonnaire, Julie and Caucheteux, Charlotte
            and D{\'e}fossez, Alexandre and Desbordes, Th{\'e}o
            and Diego-Sim{\'o}n, Pablo and Khanna, Shubh and Millet, Juliette
            and Orhan, Pierre and Panchavati, Saarang and Ratouchniak, Antoine
            and Thual, Alexis and Brooks, Teon L. and Begany, Katelyn
            and Benchetrit, Yohann and Careil, Marl{\`e}ne and Banville, Hubert
            and d'Ascoli, St{\'e}phane and Dahan, Simon and Rapin, J{\'e}r{\'e}my},
  year   = {2026},
  url    = {https://kingjr.github.io/files/neuralset.pdf},
  note   = {Preprint; URL will be updated when the paper lands on arXiv}
}
```

## Third-Party Content

References to third-party content from other locations are subject to
their own licenses and you may have other legal obligations or
restrictions that govern your use of that content.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
