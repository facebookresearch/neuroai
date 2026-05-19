.. neuralbench documentation master file, created by
   sphinx-quickstart on Thu Nov  6 20:48:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <p class="landing-badge landing-badge--hero"><i class="fas fa-flask"></i><span>NeuralBench</span></p>

NeuralBench
===========

NeuralBench is a unified framework to benchmark **NeuroAI models**. It provides
a comprehensive suite of downstream tasks for evaluating pretrained brain models,
task-specific architectures, and pretraining strategies across EEG, MEG, and fMRI.

The first release ships two EEG suites -- ``NeuralBench-EEG-Core v1.0``
(one dataset per task across the 36 EEG tasks) and ``NeuralBench-EEG-Full v1.0``
(all 94 datasets registered for the same 36 tasks) -- evaluated with
8 task-specific and 6 foundation model architectures.

.. raw:: html

   <div id="neuralbench-results-table"></div>

.. raw:: html
   :file: ../_static/neuralbench-parallel-coords.html

.. code-block:: bash

   pip install neuralbench

See :doc:`install` for more options.

----

🚀 Quickstart
---------------

Evaluate a model on a downstream task in three commands:

.. code-block:: bash

   neuralbench eeg audiovisual_stimulus --download    # 1. Download data
   neuralbench eeg audiovisual_stimulus --prepare     # 2. Prepare preprocessing cache
   neuralbench eeg audiovisual_stimulus               # 3. Run the task

The ``audiovisual_stimulus`` task uses the single-subject
`MNE-Python sample dataset <https://mne.tools/stable/documentation/datasets.html#sample>`_
(~1.5 GB).  We use it both as a quick sanity-check task and as a probe of
model behaviour in very-low-data regimes (288 trials, 4-class
classification).  Add ``--debug`` for a fast local run.

After experiments complete, re-run with ``--plot-cached`` to generate
comparison figures and tables from the stored results (no retraining):

.. code-block:: bash

   neuralbench eeg audiovisual_stimulus --plot-cached

.. admonition:: Learn more
   :class: tip

   :doc:`Open the quickstart tutorial <auto_examples/quickstart/01_run_first_task>`
   for a full walkthrough of the CLI, config system, and model selection.

----

📖 Tutorials
--------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`rocket` Quickstart
      :link: auto_examples/quickstart/01_run_first_task
      :link-type: doc
      :class-card: sd-shadow-sm

      Run your first benchmark task: CLI usage, debug mode, model
      switching, and hyperparameter grids.

   .. grid-item-card:: :fas:`chart-bar` Visualizing Results
      :link: auto_examples/results/plot_visualize_results
      :link-type: doc
      :class-card: sd-shadow-sm

      Use ``BenchmarkAggregator`` to collect results and produce
      comparison plots, results tables, and rank tables.

   .. grid-item-card:: :fas:`plus-circle` Adding a New Task
      :link: auto_examples/adding_task/create_new_task
      :link-type: doc
      :class-card: sd-shadow-sm

      Create a ``config.yaml`` for a new downstream task.

   .. grid-item-card:: :fas:`cube` Adding a New Model
      :link: auto_examples/adding_model/create_new_model
      :link-type: doc
      :class-card: sd-shadow-sm

      Register a task-specific or foundation model via a YAML config file.

   .. grid-item-card:: :fas:`wrench` Advanced
      :link: auto_examples/advanced/modify_training_loop
      :link-type: doc
      :class-card: sd-shadow-sm

      Customize the training loop by subclassing ``BrainModule``.

   .. grid-item-card:: :fas:`trophy` EEG Foundation Challenge 2025
      :link: auto_examples/eeg_challenge/plot_eeg_challenge_2025
      :link-type: doc
      :class-card: sd-shadow-sm

      Reproduce the NeurIPS 2025 challenge tasks (cross-task reaction
      time and externalizing-factor prediction) with NeuralBench.

----

Running the full EEG benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate a model across **all 36 EEG tasks**, use the ``all`` keyword:

.. code-block:: bash

   neuralbench eeg all --download    # 1. Download all automatically downloaded datasets (~3.3 TB)
   neuralbench eeg all --prepare     # 2. Build preprocessing cache for task-specific models
   neuralbench eeg all               # 3. Run all 36 EEG tasks

.. admonition:: Full guide
   :class: tip

   See :doc:`full_benchmark` for prerequisites, resource requirements,
   model and dataset variant options, and computational considerations.

----

Supported downstream tasks and models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`list` Downstream Tasks
      :link: tasks/tasks
      :link-type: doc
      :class-card: sd-shadow-sm

      Browse the full list of supported tasks with descriptions,
      datasets, and configurations.

   .. grid-item-card:: :fas:`brain` Models
      :link: models/models
      :link-type: doc
      :class-card: sd-shadow-sm

      See all available task-specific and foundation models for evaluation.

----

Citation
--------

If you use NeuralBench in your research, please cite the
`NeuralBench white paper <https://arxiv.org/abs/2605.08495>`_:

.. code-block:: bibtex

   @article{banville2026neuralbench,
     title   = {NeuralBench: A Unifying Framework to Benchmark NeuroAI Models},
     author  = {Banville, Hubert and d'Ascoli, St{\'e}phane and Dahan, Simon
                and Rapin, J{\'e}r{\'e}my and Careil, Marl{\`e}ne
                and Benchetrit, Yohann and L{\'e}vy, Jarod
                and Panchavati, Saarang and Ratouchniak, Antoine
                and Zhang, Mingfang and Cascardi, Elisa and Begany, Katelyn
                and Brooks, Teon and King, Jean-R{\'e}mi},
     year    = {2026},
     journal = {arXiv preprint arXiv:2605.08495},
     url     = {https://arxiv.org/abs/2605.08495},
   }

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   auto_examples/quickstart/index
   auto_examples/results/index
   auto_examples/adding_task/index
   auto_examples/adding_model/index
   auto_examples/advanced/index
   auto_examples/eeg_challenge/index

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Reference

    full_benchmark
    api
    tasks/tasks
    models/models
