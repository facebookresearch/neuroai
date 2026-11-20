"""
Using the Python API
=====================

:func:`~neuralbench.run_benchmark` launches the same experiments as the
:doc:`CLI </neuralbench/auto_examples/quickstart/01_run_first_task>` -- same
YAML configs, same selections, same cache -- from a script or notebook.  Reach
for it when a benchmark run is one step of a larger workflow you are scripting.

Two things it is not.  It takes a model *name*, so a model of your own goes
through :doc:`evaluate_model
</neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` instead.
And it launches runs rather than collecting them, which the :doc:`results
tutorial </neuralbench/auto_examples/results/plot_visualize_results>` covers.
"""

# %%
# Available tasks and models
# --------------------------
#
# The same discovery helpers used by the CLI are available as module-level
# constants:
#
# .. code-block:: python
#
#    from neuralbench import cli
#
#    print("Devices:", cli.ALL_DEVICES)
#    print("EEG tasks:", ", ".join(cli.TASKS["eeg"]))
#    print("Models:", ", ".join(cli.ALL_MODELS))
#
# See the :doc:`task index </neuralbench/tasks/tasks>` and
# :doc:`model index </neuralbench/models/models>` for the full lists.

# %%
# Running a single task
# ---------------------
#
# The minimal call needs only a **device** and a **task**.  Pass
# ``debug=True`` to run locally with a reduced config (2 epochs,
# 5 batches) — ideal for development.
#
# .. code-block:: python
#
#    from neuralbench import run_benchmark
#
#    run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        debug=True,
#    )
#

# %%
# What comes back
# ---------------
#
# Nothing, in every mode but one: the call launches experiments and returns an
# empty list, the results being written to the results folder under each
# experiment's UID.  Two ways to get them in hand:
#
# - ``plot_cached=True`` returns the cached results it plotted, and runs
#   nothing.  This is the CLI's ``--plot-cached``.
# - ``BenchmarkAggregator`` collects them directly, which is what the
#   :doc:`results tutorial
#   </neuralbench/auto_examples/results/plot_visualize_results>` uses for
#   comparison plots and tables.
#
# .. code-block:: python
#
#    results = run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        plot_cached=True,
#    )
#
# :func:`~neuralbench.evaluate_model` is the entry point that returns results
# from the runs it launched, as a frame.
#

# %%
# Specifying a model
# ------------------
#
# Override the default model (EEGNet) with the ``model`` parameter.
# Any name from ``ALL_MODELS`` works, as well as the group aliases
# ``"all_classic"`` and ``"all_fm"``.
#
# .. code-block:: python
#
#    results = run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        model="deep4net",
#        debug=True,
#    )
#

# %%
# Multiple tasks and models
# -------------------------
#
# Pass lists to sweep over several tasks and/or models.  Each
# combination produces a separate experiment.
#
# .. code-block:: python
#
#    run_benchmark(
#        device="eeg",
#        task=["audiovisual_stimulus", "sex"],
#        model=["eegnet", "deep4net"],
#        debug=True,
#    )
#

# %%
# Sweeping seeds
# --------------
#
# Seeds come from the grid rather than from a parameter: ``grid=True`` expands
# the task's ``grid.yaml`` on top of ``defaults/grid.yaml``, whose default
# sweep is the seed list ``[33, 34, 35]``.  That gives one experiment per seed,
# which is what a variance estimate needs.
#
# .. code-block:: python
#
#    run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        model="reve",
#        grid=True,
#    )
#

# %%
# Comparison with the CLI
# -----------------------
#
# Every ``run_benchmark()`` call has an equivalent CLI invocation:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 50 50
#
#    * - Python
#      - CLI
#    * - ``run_benchmark(device="eeg", task="audiovisual_stimulus", debug=True)``
#      - ``neuralbench eeg audiovisual_stimulus -d``
#    * - ``run_benchmark(device="eeg", task="all", model="all_classic")``
#      - ``neuralbench eeg all -m all_classic``
#    * - ``run_benchmark(device="eeg", task="sex", model="reve", grid=True)``
#      - ``neuralbench eeg sex -m reve -g``
