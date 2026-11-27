LUNA
====

| **Config name**: ``NtLuna``
| **Pretrained checkpoint**: ``thorir/LUNA`` (HuggingFace Hub, manual download)
| **Reference**: Döner et al., *LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis*, NeurIPS 2025

LUNA is a topology-invariant EEG foundation model that processes signals from
varying numbers of channels using a learned channel-unification mechanism based
on cross-attention queries.  It is **topology-agnostic**: it accepts arbitrary
montages without requiring a fixed channel set.  Three pretrained checkpoints
are available (Base, Large, Huge); the NeuralBench default uses the **Large**
variant.  The model was pretrained on the Temple University Hospital EEG corpus
(TUEG) plus the Siena Scalp EEG Database at 256 Hz.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   LUNA was pretrained on TUEG plus the Siena Scalp EEG Database (Section 4.1
   of the paper).  The paper explicitly excluded **TUAB**, **TUAR**, **TUSL**,
   and **SEED-V** from pretraining, but **did not exclude TUEV**.  This means
   ``clinical_event`` has a direct overlap, while ``pathology`` and
   ``artifact`` have corpus-level overlap (same TUEG parent corpus, but the
   specific recordings were excluded).

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Pretraining dataset
     - NeuralBench task
     - Overlap type
     - NeuralBench study
   * - TUEV (not excluded)
     - ``clinical_event``
     - direct
     - Harati2015
   * - TUEG (TUAB excluded)
     - ``pathology``
     - corpus
     - Lopez2017
   * - TUEG (TUAR excluded)
     - ``artifact``
     - corpus
     - Hamid2020

Known limitations
~~~~~~~~~~~~~~~~~

* **Patch size padding** -- ``n_times`` is automatically zero-padded to the
  next multiple of ``patch_size`` (40 at 256 Hz) when needed.  This is
  handled transparently by ``_LunaEncoderWrapper`` but may introduce a
  small number of padding tokens at the end of the sequence.
