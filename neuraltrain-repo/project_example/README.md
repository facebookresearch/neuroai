# `neuraltrain` project example

This directory contains an example project showing how to use `neuraltrain` and `pytorch-lightning` to train a neural network on MEG data.

## Description

This example loads the [MNE sample dataset](https://mne.tools/stable/documentation/datasets.html#sample) which contains MEG data from a single subject and trains a simple [ConvNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) to perform 4-class classification (left auditory, right auditory, left visual, right visual). The example grid compares different architecture and optimization configurations.

## Running the example

**1. Install neuraltrain**

See the [README](../README.md) for installation instructions.

**2. [Optional] Set up Weights & Biases**

[General instructions](https://docs.wandb.ai/quickstart)

**3. Run local example**

```
python -m project_example.grids.defaults
```

**4. Run example grid**

```
python -m project_example.grids.run_grid
```

**5. Monitor training and inspect results**

Head over to *Weights & Biases* to monitor training and inspect results.

See [`plot_figure.py`](./plot_figure.py) for an example of how to load results locally and plot them with `matplotlib`.
