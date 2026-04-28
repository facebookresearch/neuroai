(neuralfetch-contributing)=
# Contributing: Onboarding a New Study

This guide walks you through adding a new study (dataset) to
**NeuralFetch** — the curated catalog of public brain datasets used by
NeuralSet.

```{tip}
For general contributing guidelines (fork/PR workflow, pre-commit hooks,
code quality), see the [NeuralSet contributing guide](../neuralset/extending/contributing.md).
```

## Overview

Adding a study involves four steps:

1. **Upload your data** to a public repository supported by NeuralFetch.
2. **Create the study class** following the NeuralFetch format.
3. **Validate and test** the study locally.
4. **Open a Pull Request** for review.

----

## 1. Upload Data

NeuralFetch supports downloading from many public data repositories.
Choose the one that best fits your data:

| Repository | Best for | Link |
|------------|----------|------|
| **OpenNeuro** | BIDS-formatted EEG/MEG/fMRI | [openneuro.org](https://openneuro.org) |
| **HuggingFace Hub** | Any modality, large files | [huggingface.co/datasets](https://huggingface.co/datasets) |
| **OSF** | Multi-file datasets, any format | [osf.io](https://osf.io) |
| **Zenodo** | Versioned archives, DOI minting | [zenodo.org](https://zenodo.org) |
| **Dryad** | Tabular + small neural data | [datadryad.org](https://datadryad.org) |
| **DANDI** | NWB-formatted neurophysiology | [dandiarchive.org](https://dandiarchive.org) |
| **PhysioNet** | Physiological signals (EEG, ECG, EMG) | [physionet.org](https://physionet.org) |

**Key requirements:**
- Data must be **publicly accessible** (or at least requestable with a free account).
- Include a clear **licence** (CC-BY-4.0, CC0-1.0, etc.).
- Stimulus files should be included or separately downloadable.

----

## 2. Create the Study Class

Each study lives in a single Python file inside `neuralfetch-repo/neuralfetch/studies/`.

### Naming Convention

- **Class name** = PascalCase of the Google Scholar BibTeX key.
  Search your paper on [Google Scholar](https://scholar.google.com), click
  "Cite" → "BibTeX", and use the key verbatim.
  E.g., `allen2022massive` → `Allen2022Massive`.
- **File name** = lowercase class name, e.g., `allen2022massive.py`.

### Minimal Template

```python
import logging
import typing as tp

import pandas as pd
from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class AuthorYearKeyword(study.Study):
    """DatasetName: <modality> responses to <stimulus description>.

    Brief summary (2-4 sentences). State population if not healthy adults.

    Experimental Design:
        - <Device info>
        - N participants
        - Paradigm: <brief description>
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ()
    bibtex: tp.ClassVar[str] = """
    @article{authoryearkeyword,
        title={...},
        author={...},
        journal={...},
        year={YYYY},
        doi={10.xxxx/...}
    }
    """
    url: tp.ClassVar[str] = "https://..."
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings from N participants during task."
    )

    _info: tp.ClassVar[study.StudyInfo] = None  # populated after validation

    def _download(self) -> None:
        # Each repository has a matching download class (Openneuro, Osf,
        # Physionet, Huggingface, …) — see the API reference.
        dl = download.Openneuro(study="dsXXXXXX", dset_dir=self.path / "download")
        dl.download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for sub_dir in sorted((self.path / "download").glob("sub-*")):
            yield {"subject": sub_dir.name}

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        # Load events for a single subject/session
        ...
```

### ClassVar Order

Always declare class variables in this order:
`aliases` → `bibtex` → `url` → `licence` → `description` → `requirements` → `_info`

----

## 3. Validate and Test

### Run the study locally

```python
import neuralset as ns

study = ns.Study(name="AuthorYearKeyword", path="./test_data")
study.download()
events = study.run()

# Check basic properties
print(f"Events: {len(events)}")
print(f"Subjects: {events['subject'].nunique()}")
print(f"Event types: {events['type'].value_counts()}")
```

### Compute study info

Once the data is downloaded and the study runs correctly, compute the
`StudyInfo` metadata:

```python
from neuralfetch.utils import compute_study_info

info = compute_study_info("AuthorYearKeyword", study.path)
print(info)
# Copy this into _info in your class
```

### Run the test suite

```bash
cd neuralfetch-repo
pytest neuralfetch/ -x -q -k "AuthorYearKeyword"
```

The `test_study_info` test only runs for studies where `_info is not None`,
so populating `_info` is what enables test coverage.

----

## 4. Open a Pull Request

1. **Branch** from the current dev branch:
   ```bash
   git checkout -b add-authoryearkeyword
   ```
2. **Add your file** to `neuralfetch-repo/neuralfetch/studies/`.
3. **Run pre-commit hooks**:
   ```bash
   pre-commit run --all-files
   ```
4. **Push** and open a PR. Include in the description:
   - Link to the dataset and paper.
   - Number of subjects and event types.
   - Any known issues or caveats.

----

## Useful References

- [Create Your Own Study tutorial](auto_examples/03_create_new_study) —
  step-by-step Sphinx Gallery tutorial with runnable code.
- [NeuralSet contributing guide](../neuralset/extending/contributing.md) —
  general coding standards, pre-commit hooks, fork/PR workflow.
- [NeuralFetch download helpers](reference/reference) —
  API reference for `download.Openneuro`, `download.Osf`,
  and all other download backends.

```{raw} html
<div class="page-nav">
  <a href="samples.html" class="page-nav-btn page-nav-btn--outline">&larr; Sample Datasets</a>
</div>
```
