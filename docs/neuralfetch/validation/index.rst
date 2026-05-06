Study Validation Reports
========================

Validation reports are **decoding/encoding feasibility checks** for each
study in the catalogue.  For every study with a config under
``neuralfetch/validations/``, the runner executes one representative
:class:`~neuralyze.SlidingWindow` analysis — and optionally a
:class:`~neuralyze.trf.TRFScoring` encoding analysis — and generates an
interactive MNE Report so we can confirm that:

- the full data pipeline (raw -> events -> neuro -> features -> model
  -> time-resolved scores) runs end-to-end on the dataset,
- meaningful signal can be decoded (or encoded) from the data at a
  sensible time course, and
- per-subject and per-channel quality look reasonable (drop grid,
  ERP/ERF, peak-score table).

This is **not** a full replication of each paper's analysis.  A config
may optionally set a ``reference_metric`` (and ``reference_metric_name``)
to overlay a published number on the decoding plots for context, but
matching a paper is not a requirement -- most configs only exercise the
default pipeline.

.. rubric:: How to run a validation

Study names are matched **case-insensitively**, so both
``Grootswagers2022Human`` and ``grootswagers2022human`` resolve correctly.

.. code-block:: bash

   # Generate an MNE Report for a specific study:
   python -m neuralfetch validate Grootswagers2022Human \
       --output-dir docs/neuralfetch/validation/reports

   # List all studies with a validation config:
   python -m neuralfetch validate --list

.. rubric:: Available reports

Reports are interactive MNE Report HTML files in ``reports/``.  Each
report bundles:

- a **Study Information** block (package versions, study metadata,
  dataset summary, echoed analysis config including TRF config when set,
  BibTeX citation),
- a **Grand Average** time-resolved score curve,
- a **Group Grand Average** clickable plot with per-subject toggles,
- a **TRF Grand Average** per-channel encoding score bar chart (only when
  a ``trf`` config is set),
- a **Participants x Channels** drop grid for quick QC of bad channels
  and excluded subjects,
- a **Results Summary** table of peak score and peak time per subject,
- **per-subject** score curves paired with ERP / ERF / Evoked
  ``plot_joint`` figures,
- **per-subject TRF** figures (only when a ``trf`` config is set), each
  containing a per-channel bar chart and, when sensor positions are
  available, a topomap of encoding scores via
  :func:`mne.viz.plot_topomap`, and
- an optional reference-metric line overlay when ``reference_metric``
  is set in the config.

Once generated, the reports are served as static HTML alongside these
docs.  See the :doc:`studies catalogue </brainai/explore>` for a
quick view of which studies have a validation report.

.. rubric:: Validation configuration

Each study config is a TOML file in ``neuralfetch/validations/``.  The
file is parsed into a :class:`~neuralfetch.utils.validation.StudyValidation`
instance at load time — no Python boilerplate required.  A minimal example:

.. code-block:: toml

   study_name = "MyStudy2025"
   description = "Decode word embeddings from MEG"
   event_type  = "Word"
   start = -0.1
   stop  = 1.0

   [neuro]
   name      = "MegExtractor"
   frequency = 100

   [extractor]
   name        = "WordLength"
   aggregation = "trigger"

   [model.sklearn_model]
   name = "RidgeCV"

   [cv]
   name     = "Cv"
   n_splits = 5

   [scoring]
   name = "corr"

The key fields are:

- **neuro / extractor / model / cv / scoring** — passed directly to
  :class:`~neuralyze.SlidingWindow`.
- **trf** — optional TOML table that enables a
  :class:`~neuralyze.trf.TRFScoring` encoding run in addition to the
  sliding-window decoding; parsed into a
  :class:`~neuralfetch.utils.validation.TRFConfig`.  Key sub-fields:

  .. code-block:: toml

     [trf]
     tmin        = 0.0
     tmax        = 0.3
     aggregation = "sum"
     n_pca       = 0.95

  - ``tmin`` / ``tmax`` — lag window in seconds.
  - ``aggregation`` — how features are placed on the continuous
    time-series (``"sum"`` for discrete image/word paradigms).
  - ``n_pca`` — PCA reduction applied to stimulus features before
    fitting the TRF.  A float (e.g. ``0.95``) keeps enough components
    to explain that fraction of variance; an int sets an exact count.
    **Required for high-dimensional feature spaces** such as DINOv2
    (384–768 dims) where the full delay matrix would exhaust RAM.
    All image configs use ``0.95``.  Omit the key to disable PCA.

- **infra** — optional TOML table of default resource parameters so the
  exact command needed to reproduce the run is self-documenting.  CLI
  flags always override these defaults.

  .. code-block:: toml

     [infra]
     cluster         = "slurm"
     slurm_partition = "learnfair"
     timeout_min     = 480
     mem_gb          = 128
     cpus_per_task   = 10

.. rubric:: Image studies

All image-based validation configs use
:class:`~neuralyze.HuggingFaceImage` with
``model_name = "facebook/dinov2-small"`` as the default feature extractor.
To experiment with a different model, pass ``--extractor.model_name``
on the CLI or edit the ``extractor`` table in the TOML file.
