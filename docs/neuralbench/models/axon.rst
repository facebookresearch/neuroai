AXON
====

| **Config name**: ``axon``
| **Pretrained checkpoint**: ``MannasAI/axon-eeg`` (HuggingFace Hub, Apache-2.0)
| **Reference**: Jain et al., *Adaptive Anisotropic Attention for Axis-Structured Signals*, arXiv:2609.08788, 2026

AXON (**AX**\ is-factorized **O**\ perator **N**\ etwork) is a 118.6M-parameter
transformer encoder for EEG, pretrained with masked autoencoding at 200 Hz.
Each window is cut into one token per electrode per one-second patch (0.9 s
stride), giving a grid of electrodes by time steps.  Every layer runs a
temporal attention path (each electrode across time) and a spatial attention
path (all electrodes at one time step) in parallel, mixed by a learned
per-token gate.  Electrodes are identified only by their 3-D scalp position,
so the model accepts any montage and channel order.

Channel placement
~~~~~~~~~~~~~~~~~

NeuralBench provides channel names, not positions.  ``NtAxon`` looks each name
up in MNE's ``standard_1005`` montage in head coordinates, extended with the
EGI ``GSN-HydroCel-129`` names (``E1`` ... ``E128``).  The lookup ignores case,
strips an ``EEG`` prefix or ``-REF`` suffix, maps legacy aliases (``T3`` ->
``T7``) and places a bipolar ``A-B`` channel at the midpoint of its two
electrodes.  Channels that still cannot be placed are dropped with a warning;
``channel_mapping`` (shared with REVE) renames systems such as Neuromag
``EEG 0xx`` first.

Downstream setup
~~~~~~~~~~~~~~~~

The encoder returns a ``(batch, channels, patches, 512)`` token grid.  The
default config averages it over channels and patches (AXON's own pooling) and
trains a linear head with full fine-tuning, as for the other foundation
models.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   AXON was pretrained on the Temple University Hospital EEG Corpus (TUEG),
   the I-CARE post-cardiac-arrest EEG database and private clinical
   recordings.  TUEG encompasses **TUAB**, **TUAR** and **TUEV**, which are
   also NeuralBench evaluation datasets, so results on the affected tasks may
   be inflated.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Pretraining dataset
     - NeuralBench task
     - NeuralBench study
   * - TUAB (subset of TUEG)
     - ``pathology``
     - Lopez2017
   * - TUAR (subset of TUEG)
     - ``artifact``
     - Hamid2020
   * - TUEV (subset of TUEG)
     - ``clinical_event``
     - Harati2015

Known limitations
~~~~~~~~~~~~~~~~~

* **Requires** ``braindecode >= 2.0`` (the first release that ships AXON).
* **Positions come from a template montage** --- digitised electrode positions
  in the recordings are not used; every channel is placed at its standard
  10-05 (or HydroCel) location.
