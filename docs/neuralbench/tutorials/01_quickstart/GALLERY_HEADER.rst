Quickstart
==========

Run your first neuralbench evaluation in a few commands.

Which entry point?
------------------

The same experiments are reachable three ways -- same YAML configs, same cache,
same results folder.  What decides between them is where your model lives and
what you want handed back.

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * -
     - ``neuralbench`` CLI
     - ``run_benchmark()``
     - ``evaluate_model()``
   * - Names the model
     - registered name (``-m reve``)
     - registered name (``model="reve"``)
     - an ``nn.Module`` you built
   * - Covers
     - tasks x datasets x models x grid
     - the same selections
     - one instance, over tasks and datasets
   * - Hands back
     - results on disk; ``--plot-cached`` for figures and tables
     - the same, from Python
     - the results, as a ``DataFrame``
   * - Runs on
     - Slurm, or locally with ``--debug``
     - the same
     - this process, or Slurm with ``cluster="auto"``

Reach for the CLI to run or reproduce the benchmark, for
:func:`~neuralbench.run_benchmark` to drive those same runs from a script, and
for :func:`~neuralbench.evaluate_model` while developing a model that has no
config in this repo.  The three tutorials below follow that order.

.. note::
   📥 **You can download these tutorials with the buttons at the bottom of each page.**
