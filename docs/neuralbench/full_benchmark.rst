.. _full_eeg_benchmark:

Running the full EEG benchmark
================================

This page describes how to run the **entire** benchmark of EEG tasks end-to-end,
including prerequisites, step-by-step instructions, model and dataset options,
and computational resource requirements.

Prerequisites
-------------

- **SLURM cluster with GPUs** (or a single machine with a GPU for ``--debug`` mode).
- **Disk space**: ~3.3 TB for the raw datasets of the 35 base EEG tasks
  (~4.4 TB when including all multi-dataset variants via ``--dataset all``).
  An additional ~35 GB is needed for the preprocessing cache.
- **Data access**: a handful of datasets cannot be fetched automatically and
  must be obtained by registering, signing a license agreement, or filling
  out an application form. See :ref:`manual-download-datasets` below for
  the full list and step-by-step instructions.

.. _manual-download-datasets:

Datasets requiring manual download
-----------------------------------

Most NeuralBench datasets are downloaded automatically by ``neuralbench
<device> <task> --download``. The following datasets are *exceptions*: they
require a one-time manual step (creating an account, accepting a license,
or submitting an application form) before they can be obtained. Tasks that
depend on these datasets will be skipped with a warning during
``neuralbench eeg all --download``.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Task(s)
     - Dataset / Study
     - Manual step required
   * - ``eeg/pathology``, ``eeg/artifact``, ``eeg/clinical_event``
     - TUH EEG Corpus (``Lopez2017Tuab``, ``Hamid2020Tuar``,
       ``Harati2015Tuev``)
     - Submit an access request to Temple University NEDC at
       https://isip.piconepress.com/projects/nedc/html/tuh_eeg/. Once
       granted, place the ``tuh_eeg`` tree under ``DATA_DIR`` so that the
       symlinked subfolders (``tuab``, ``tuar``, ``tuev``) are discoverable.
   * - ``eeg/image``, ``meg/image``
     - THINGS-images database (shared by ``Gifford2022Large`` and
       ``Hebart2023ThingsMeg``)
     - Read and accept the THINGS license at
       https://osf.io/jum2f/files/52wrx, retrieve the password from
       ``password_images.txt``, and export it as
       ``NEURALFETCH_THINGS_PASSWORD`` (used to unzip the ~12 GB
       ``images_THINGS.zip`` archive).
   * - ``eeg/emotion``
     - FACED (``Chen2023Large``, hosted on Synapse)
     - Create a Synapse account at https://www.synapse.org/, accept the
       Synapse data-use terms, generate a Personal Access Token with
       *View* and *Download* scopes, and export it as
       ``NEURALFETCH_SYNAPSE_TOKEN``.
   * - ``eeg/video``
     - SEED-DV (``Liu2024Eeg2video``)
     - Sign the BCMI Lab license agreement
       (https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc), submit the
       application form at
       https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ (select
       *SEED-DV*, use an institutional email), and download from the link
       sent by email after approval.
   * - ``fmri/image``
     - Natural Scenes Dataset (``Allen2022Nsd``)
     - Run ``--download`` from an interactive terminal: NeuralBench will
       display the NSD Terms and Conditions
       (https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions),
       collect the user information required by the NSD Data Access
       Agreement, and submit the agreement form on your behalf. The AWS
       CLI must also be installed in order to ``aws s3 sync`` the dataset
       (~several TB).
   * - ``eeg/motor_imagery``, ``eeg/mental_arithmetic``
     - Shin2017OpenA, Shin2017OpenB (via MOABB)
     - Read and accept the GNU General Public License v3 terms at
       http://doc.ml.tu-berlin.de/hBCI, then export the environment
       variable ``MOABB_ACCEPT_LICENCE=1`` to allow downloading.
   * - ``eeg/typing``, ``meg/typing``
     - ``Levy2025BrainEeg``, ``Levy2025BrainMeg`` (formerly
       ``Pinet2024Eeg``/``Pinet2024Meg``)
     - Not yet available for public download. The release timeline is
       tracked alongside the upcoming dataset paper; until then these
       tasks remain a documented known limitation.
   * - ``eeg/speech``
     - Brennan2019 (``Brennan2019Hierarchical``) on Deep Blue Data
     - Auto-download via ``urlretrieve`` against
       ``deepblue.lib.umich.edu/data/downloads/<id>`` is currently
       blocked by Cloudflare's managed challenge (HTTP 403; see
       ``scratch/hubertjb/probe_brennan2019_url.sh`` for the diagnostic
       probe). Until upstream restores anonymous HTTP access, fetch the
       **v1** files manually via Globus: create a free Globus account
       (https://www.globus.org), open the dataset page at
       https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t,
       click *Files → Transfer files using Globus*, and transfer the
       full ``v1/`` directory to
       ``$DATA_DIR/Brennan2019Hierarchical/download/`` (~4.2 GB, 56
       files: 33 ``S*.mat``, ``audio.zip``, ``proc.zip``,
       ``AliceChapterOne-EEG.csv``, ``README.txt``). Once present,
       re-running ``neuralbench eeg speech --download`` will skip the
       fetch step and proceed.

In addition, a few MOABB-backed datasets (e.g. ``Shin2017OpenA`` /
``Shin2017OpenB`` used by ``eeg/mental_arithmetic``) require accepting a
GPL-style click-through license. Set ``MOABB_ACCEPT_LICENCE=1`` in your
environment to acknowledge it before running ``--download``.

Step 1: Download
----------------

Download all datasets for the 36 EEG tasks:

.. code-block:: bash

   neuralbench eeg all --download

This triggers the download of all required studies to ``DATA_DIR``. Tasks whose
datasets require manual access (see :ref:`manual-download-datasets` above) will
be skipped with a warning until the corresponding credentials, password, or
data-use agreement are in place.

Step 2: Prepare the cache
--------------------------

Preprocess and cache the data for all tasks:

.. code-block:: bash

   neuralbench eeg all --prepare

Each task is submitted as a SLURM job (or run locally with ``--debug``) that
loads the raw data, applies the preprocessing pipeline (resampling, filtering,
scaling), and writes the result to ``CACHE_DIR``. This step can be parallelized
across tasks.

Step 3: Run the benchmark
--------------------------

Launch the full benchmark with the default model (EEGNet):

.. code-block:: bash

   neuralbench eeg all

This submits 3 SLURM jobs per task (108 jobs total for all 36 tasks, one per
seed). Each job trains, validates with early stopping, and evaluates the model.

Running with different models
-----------------------------

Use ``-m`` to specify alternative models or model groups:

.. code-block:: bash

   neuralbench eeg all -m eegconformer             # Single model
   neuralbench eeg all -m all_classic               # All 8 task-specific models
   neuralbench eeg all -m all_fm                    # All 6 foundation models
   neuralbench eeg all -m all                       # All models

**Task-specific models** (8): ``shallow_fbcsp_net``, ``simpleconv_time_agg``,
``eegnet``, ``deep4net``, ``eegconformer``, ``atcnet``, ``bdtcn``, ``ctnet``

**Foundation models** (6): ``bendr``, ``biot``, ``cbramod``, ``labram``,
``luna``, ``reve``

Running with dataset variants
------------------------------

Nine tasks support multiple datasets. Use ``--dataset all`` to evaluate across
all dataset variants for a task:

.. code-block:: bash

   neuralbench eeg motor_imagery --dataset all      # Run on all 18 motor imagery datasets
   neuralbench eeg p3 --dataset all                 # Run on all 24 P300 datasets

Visualizing results
--------------------

Results can be visualized on Weights & Biases, or aggregated locally using
``--plot-cached``:

.. code-block:: bash

   neuralbench eeg all -m all_classic all_fm --plot-cached

``--plot-cached`` does not re-train any model. It collects the stored test
metrics from the cache and produces the following outputs in the
``outputs/`` directory, split into three subfolders (``core/`` for the
``NeuralBench-EEG-Core v1.0`` plots/tables (one dataset per task), ``full/``
for the ``NeuralBench-EEG-Full v1.0`` per-dataset breakdowns and variability
analyses, and ``other/`` for everything else):

- **Bar chart** (``outputs/core/core_bar_chart.png``): faceted bar chart with
  one panel per task, one bar per model, including error bars and individual
  data points.
- **Results table** (``outputs/core/core_results_table.csv``): wide-format
  table with ``mean +/- std`` per task and model.
- **Rank table** (``outputs/core/core_rank_table.csv``): models ranked within
  each task (1 = best), with an average rank row at the bottom.

See the
:doc:`Visualizing Results <auto_examples/results/plot_visualize_results>`
tutorial for details on ``BenchmarkAggregator`` configuration and the
available visualization methods.

Evaluation protocol
--------------------

Each model is evaluated **3 times** per task with different model seeds
(33, 34, 35). The train/val/test data splits are **fixed** across all runs,
so variance reflects model training stochasticity only. Reported metrics are
**mean +/- std** over the 3 runs. See individual
:doc:`task pages <tasks/tasks>` for split details.

Computational considerations
-----------------------------

NeuralBench is designed to run on a **SLURM cluster with GPUs**. For **local
development** without SLURM, use ``--debug`` mode, which runs on a single GPU
with a subsampled dataset (2 epochs, 5 batches per epoch).

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Resource
     - Default / Requirement
   * - GPU
     - 1 x volta32gb (32 GB VRAM)
   * - CPU RAM per job
     - 64 GB
   * - Raw datasets (35 base EEG tasks)
     - ~3.3 TB
   * - Preprocessing cache
     - ~35 GB
