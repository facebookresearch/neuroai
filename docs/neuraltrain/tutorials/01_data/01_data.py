# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Pipeline
=============

The ``Data`` stage is where ``neuraltrain`` connects to ``neuralset``.
You start from a study chain plus a ``Segmenter``, then produce
train/val/test PyTorch loaders for the rest of the training pipeline.

This tutorial covers:

- what the ``Data`` config is responsible for
- the ``study chain → Segmenter → DataLoader`` flow
- customising split policies and extractor layouts
- worker and infrastructure settings

.. seealso::

   :doc:`/neuraltrain/project_example` -- full example project
"""

# %%
# The Data config
# ---------------
#
# The example project defines a ``Data`` pydantic model with four fields:
#
# - ``study``: a neuralset ``Step`` (usually a chain of study + split)
# - ``segmenter``: a ``ns.dataloader.Segmenter`` with extractors
#   and the segment window
# - ``batch_size`` and ``num_workers``
#
# .. code-block:: python
#
#    class Data(pydantic.BaseModel):
#        study: ns.Step
#        segmenter: ns.dataloader.Segmenter
#        batch_size: int = 64
#        num_workers: int = 0
#
# Its ``build()`` method runs the study chain, applies the segmenter,
# prepares extractors, and wraps each split into a ``DataLoader``:
#
# .. code-block:: python
#
#    def build(self) -> dict[str, DataLoader]:
#        events = self.study.run()
#        dataset = self.segmenter.apply(events)
#        dataset.prepare()
#        loaders = {}
#        for split, shuffle in [("train", True), ("val", False), ("test", False)]:
#            ds = dataset.select(dataset.triggers["split"] == split)
#            loaders[split] = DataLoader(ds, collate_fn=ds.collate_fn, ...)
#        return loaders

# %%
# The default study chain
# -----------------------
#
# In the default config, the study is a two-step chain:
#
# 1. ``Mne2013Sample`` -- loads the MNE sample dataset
# 2. ``SklearnSplit`` -- adds a ``split`` column via stratified splitting
#
# Let's look at how this is configured:

default_data_config = {
    "study": [
        {
            "name": "Mne2013Sample",
            "path": "/tmp/data/mne2013sample",
            "query": None,
        },
        {
            "name": "SklearnSplit",
            "valid_split_ratio": 0.2,
            "test_split_ratio": 0.2,
            "valid_random_state": 87,
            "split_by": "_index",
        },
    ],
    "segmenter": {
        "extractors": {
            "input": {
                "name": "MegExtractor",
                "frequency": 120.0,
                "filter": (0.5, 25.0),
                "baseline": (0.0, 0.1),
                "scaler": "RobustScaler",
                "clamp": 16.0,
            },
            "target": {
                "name": "EventField",
                "event_types": "Stimulus",
                "event_field": "code",
            },
        },
        "trigger_query": "type == 'Stimulus'",
        "start": -0.1,
        "duration": 0.5,
    },
    "batch_size": 16,
}

for section, value in default_data_config.items():
    print(f"{section}: {value}\n")

# %%
# The Segmenter
# -------------
#
# The ``Segmenter`` is the neuralset object that turns an events
# DataFrame into a ``SegmentDataset``. It holds extractors, a trigger
# query, and the segment time window. Let's instantiate one:

import neuralset as ns

segmenter = ns.dataloader.Segmenter(
    extractors={
        "input": {
            "name": "MegExtractor",
            "frequency": 120.0,
            "filter": (0.5, 25.0),
            "baseline": (0.0, 0.1),
            "scaler": "RobustScaler",
            "clamp": 16.0,
        },
        "target": {
            "name": "EventField",
            "event_types": "Stimulus",
            "event_field": "code",
        },
    },
    trigger_query="type == 'Stimulus'",
    start=-0.1,
    duration=0.5,
)

print(segmenter)

# %%
# .. raw:: html
#
#    <div class="page-nav">
#      <a href="../../tutorials.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorials</a>
#      <a href="../models/02_models.html" class="page-nav-btn">Tutorial 2: Models &rarr;</a>
#    </div>
