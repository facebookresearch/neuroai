# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retrieve results from wandb and plot with matplotlib.

To run::

    python -m project_example.plot_figure
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from exca import ConfDict

from .grids.defaults import PROJECT_NAME
from .grids.run_grid import GRID_NAME

api = wandb.Api()

all_df = []
runs = api.runs(
    path=f"{wandb.api.default_entity}/{PROJECT_NAME}", filters={"group": GRID_NAME}
)

# Extract config and results from wandb runs
for run in runs:
    hist_df = run.history(keys=["epoch", "test_acc"])
    hist_df["name"] = run.name
    flat_config = ConfDict(run.config).flat()
    config_df = pd.Series(flat_config).to_frame().T
    hist_df = pd.concat([hist_df, config_df], axis=1)

    all_df.append(hist_df)

results_df = pd.concat(all_df, axis=0, ignore_index=True)

results_df["test_acc"] *= 100
test_metric = "test_acc"
chance_level = results_df.loc[results_df.n_epochs == 0, test_metric].mean()

# Plot accuracy
plot_df = results_df[results_df.n_epochs > 0]
ax = sns.barplot(
    data=plot_df,
    x="brain_model_config.depth",
    y=test_metric,
    hue="lr",
    palette="colorblind",
)
ax.axhline(chance_level, color="k", alpha=0.5, ls="--")
ax.set_xlabel("Depth")
ax.set_ylabel("Test accuracy (%)")
ax.set_title(f"{PROJECT_NAME}, {GRID_NAME}")

# Save it
fname = f"./{PROJECT_NAME}-{GRID_NAME}_example.png"
plt.savefig(fname)
print(f"Saved figure to {fname}.")
