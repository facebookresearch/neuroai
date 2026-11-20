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

Downloads made through `download_file`, including Zenodo files, honor inherited
creation permissions on POSIX systems after the transfer and any requested
checksum verification succeed. For group-readable/writable files and directories:

```bash
umask 002
neuralfetch download MyStudy
```

The file mode follows the process umask and destination directory policy, as for
normal file creation: `002` gives `664`, `007` gives `660`, and `077` gives `600`
when no default ACL changes the result. Directories and extracted zip members
retain their normal creation behavior. Incomplete download payloads remain
private until verification; cache hits and failed transfers do not change
existing files. Correcting permissions of previously downloaded files is a
separate operation.

The implementation uses an empty, randomly named sibling permission probe; it
does not read/change the process umask or copy the downloaded payload. Permission
failures are raised before a success marker is written; the verified file is
left private and any later permission correction is a separate operation.
Platforms without POSIX permission support retain Pooch's behavior. This policy does not apply to
backends using other clients, such as S3 or NEMAR.

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
