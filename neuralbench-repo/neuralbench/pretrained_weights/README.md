# Pretrained weights for EEG foundation models

All foundation model weights are loaded from HuggingFace Hub via
braindecode's `from_pretrained()` method (requires `braindecode >= 1.4.0`).
No manual downloading is needed -- weights are fetched and cached
automatically on first use.

Make sure the HuggingFace Hub package is installed:

```bash
pip install braindecode[hub]
```

## Models and their Hub repositories

| Model | Hub repository | Notes |
|-------|---------------|-------|
| [LaBraM](https://huggingface.co/braindecode/labram-pretrained) | `braindecode/labram-pretrained` | |
| [BIOT](https://huggingface.co/braindecode/biot-pretrained-six-datasets-18chs) | `braindecode/biot-pretrained-six-datasets-18chs` | |
| [CBraMod](https://huggingface.co/braindecode/cbramod-pretrained) | `braindecode/cbramod-pretrained` | |
| [BENDR](https://huggingface.co/braindecode/braindecode-bendr) | `braindecode/braindecode-bendr` | |
| [REVE](https://huggingface.co/brain-bzh/reve-base) | `brain-bzh/reve-base` | |
| [LUNA](https://huggingface.co/PulpBio/LUNA) | `PulpBio/LUNA` | Multi-file repo, selected via `pretrained_filename` |

## [LUNA](https://huggingface.co/PulpBio/LUNA)

The pretrained weights for LUNA require manual downloading from HuggingFace. This can be done with the following:

```bash
cd neuralbench-repo/neuralbench/pretrained_weights
hf download PulpBio/LUNA --include "LUNA_*" --local-dir .
```
