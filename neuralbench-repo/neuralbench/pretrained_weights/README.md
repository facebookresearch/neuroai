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
| [REVE](https://huggingface.co/brain-bzh/reve-base) | `brain-bzh/reve-base` | Gated -- see below |
| [LUNA](https://huggingface.co/PulpBio/LUNA) | `PulpBio/LUNA` | Multi-file repo, selected via `pretrained_filename` |

## REVE -- gated access

Accessing the pretrained weights for REVE requires agreeing to data usage
terms on HuggingFace Hub.

Steps:
1. Create a HuggingFace account or log in at `https://huggingface.co`.
2. Read and accept the data usage terms on the [model repository](https://huggingface.co/collections/brain-bzh/reve).
3. Make sure the HuggingFace hub package is installed in your environment, e.g. `pip install huggingface-hub`.
4. Follow the [instructions](https://huggingface.co/docs/huggingface_hub/v0.16.3/quick-start#login) to get an access token and to log in.

The pretrained weights should now be available to load when creating the REVE model with the `from_pretrained()` method.

## [LUNA](https://huggingface.co/PulpBio/LUNA)

The pretrained weights for LUNA require manual downloading from HuggingFace. This can be done with the following:

```bash
cd neuralbench-repo/neuralbench/pretrained_weights
huggingface-cli download PulpBio/LUNA --include "LUNA_*" --local-dir .
```
