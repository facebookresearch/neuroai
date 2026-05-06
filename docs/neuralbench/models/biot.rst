BIOT
====

| **Config name**: ``biot``
| **Pretrained checkpoint**: ``braindecode/biot-pretrained-six-datasets-18chs`` (HuggingFace Hub)
| **Reference**: Yang et al., *BIOT: Biosignal Transformer for Cross-data Learning in the Wild*, NeurIPS 2023

BIOT is a transformer-based foundation model for biosignal classification.
The pretrained checkpoint was trained on six EEG datasets (PREST, SHHS, CHB-MIT,
IIIC Seizure, TUAB, TUEV) at 200 Hz using 18 bipolar channels derived from the
Temporal Central Parasagittal (TCP) montage.

In NeuralBench, a ``Conv1d`` channel projection maps from the recording's
arbitrary channel count to the 18 expected channels, allowing BIOT to be
evaluated on datasets with diverse montages without manual channel selection.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   The ``biot-pretrained-six-datasets-18chs`` checkpoint was pretrained on six
   EEG datasets including **TUAB** and **TUEV** (Section 3.6 of the paper).
   These are used both in pretraining and as NeuralBench evaluation datasets,
   so results on the affected tasks may be inflated.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Pretraining dataset
     - NeuralBench task
     - NeuralBench study
   * - TUAB
     - ``pathology``
     - Lopez2017
   * - TUEV
     - ``clinical_event``
     - Harati2015

Known limitations
~~~~~~~~~~~~~~~~~

* **Fixed spatial embeddings** -- The pretrained positional embeddings are tied
  to the 18 TCP channels.  The ``Conv1d`` channel projection learns a linear
  mapping into this space, but there is no guarantee that the resulting channels
  align with the original TCP bipolar semantics.
