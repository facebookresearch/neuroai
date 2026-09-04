"""
Using the Python API
=====================

As an alternative to the :doc:`CLI
</neuralbench/auto_examples/quickstart/01_run_first_task>`,
NeuralBench also exposes a Python function
:func:`~neuralbench.run_benchmark` that lets you launch experiments
directly from a script or notebook.  This is convenient when you want
to integrate benchmark runs into a larger workflow or inspect results
programmatically without parsing log files.
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
#    results = run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        debug=True,
#    )
#
#    print(f"Got {len(results)} result(s)")
#    for r in results:
#        print(r)
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
#    results = run_benchmark(
#        device="eeg",
#        task=["audiovisual_stimulus", "sex"],
#        model=["eegnet", "deep4net"],
#        debug=True,
#    )
#
#    print(f"Ran {len(results)} experiment(s)")
#

# %%
# Controlling random seeds
# ------------------------
#
# The ``seeds`` parameter overrides the grid's seed list (default
# ``[33]``).  Passing multiple seeds generates one experiment per
# seed, which is useful for estimating variance.
#
# .. code-block:: python
#
#    results = run_benchmark(
#        device="eeg",
#        task="audiovisual_stimulus",
#        debug=True,
#        seeds=[33, 34, 35],
#    )
#
#    print(f"Ran {len(results)} experiment(s) (one per seed)")
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
