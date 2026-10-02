"""
Visualizing Benchmark Results
==============================

Use :class:`~neuralbench.main.BenchmarkAggregator` to collect test
metrics from completed experiments and produce comparison plots and
summary tables.
"""

# %%
# How it works
# -------------
#
# ``BenchmarkAggregator`` is the built-in tool for visualizing
# results.  It wraps a list of
# :class:`~neuralbench.main.Experiment` objects and, when
# ``.run()`` is called:
#
# 1. Calls ``experiment.run()`` on each experiment to collect
#    (or retrieve from cache) the test-metric dictionary.
# 2. Merges each result dict with the experiment config into a
#    single DataFrame.
# 3. Maps each loss type to a primary metric and produces:
#
#    - A **bar chart** (``outputs/core/core_bar_chart.png``)
#    - A **results table** (``outputs/core/core_results_table.csv``)
#    - A **rank table** (``outputs/core/core_rank_table.csv``)
#
# Outputs are organised into three subfolders of ``outputs/``:
# ``core/`` (Core suite: one dataset per task -- e.g.
# ``NeuralBench-EEG-Core v1.0`` for an EEG run), ``full/`` (Full suite:
# per-dataset breakdowns + dataset-level variability -- e.g.
# ``NeuralBench-EEG-Full v1.0``), and ``other/`` (data scaling,
# computational stats, ...).

# %%
# Triggering it from the CLI
# ----------------------------
#
# The simplest way to use ``BenchmarkAggregator`` is the
# ``--plot-cached`` flag.  After experiments have been run (and
# cached), re-invoke the CLI with ``--plot-cached`` to collect the
# stored results and generate all outputs without retraining:
#
# .. code-block:: bash
#
#    # First, run experiments (results are cached automatically)
#    neuralbench eeg audiovisual_stimulus sleep_stage -m eegnet eegconformer
#
#    # Then, re-run with --plot-cached to generate comparison outputs
#    neuralbench eeg audiovisual_stimulus sleep_stage -m eegnet eegconformer --plot-cached
#
# ``--plot-cached`` does not launch any experiments; it only reads the
# results already persisted by ``exca`` and drives
# ``BenchmarkAggregator``.

# %%
# Configuration
# ---------------
#
# ``BenchmarkAggregator`` has a few configurable attributes:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 35 65
#
#    * - Field
#      - Default
#    * - ``max_workers``
#      - 256
#    * - ``collect_max_workers``
#      - 32
#    * - ``debug``
#      - False
#    * - ``output_dir``
#      - ``"<neuralbench-repo>/outputs"``

# %%
# Loss-to-metric mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``loss_to_metric_mapping`` attribute determines which metric
# is used as the **primary performance indicator** for each task
# type.  This is used for plotting and ranking:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 40 60
#
#    * - Loss
#      - Primary metric
#    * - ``CrossEntropyLoss``
#      - ``test/bal_acc``
#    * - ``BCEWithLogitsLoss``
#      - ``test/f1_score_macro``
#    * - ``MSELoss``
#      - ``test/pearsonr``
#    * - ``ClipLoss``
#      - ``test/full_retrieval/top5_acc_subject-agg``
