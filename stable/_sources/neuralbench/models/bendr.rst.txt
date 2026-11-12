BENDR
=====

| **Config name**: ``bendr``
| **Pretrained checkpoint**: ``braindecode/braindecode-bendr`` (HuggingFace Hub)
| **Reference**: Kostas et al., *BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data*, Front. Hum. Neurosci. 2021

BENDR is a wav2vec 2.0-inspired foundation model for EEG. It consists of a
convolutional encoder followed by a transformer contextualizer. The pretrained
checkpoint was self-supervised on the Temple University Hospital EEG Corpus
(TUEG v1.1/1.2) at 256 Hz using 20 channels (19 standard 10/20 EEG electrodes
+ 1 relative amplitude channel).

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   The ``braindecode-bendr`` checkpoint was pretrained on the Temple University
   Hospital EEG Corpus (TUEG), which encompasses **TUAB** and **TUEV**.  These
   are used both in pretraining and as NeuralBench evaluation datasets, so
   results on the affected tasks may be inflated.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Pretraining dataset
     - NeuralBench task
     - NeuralBench study
   * - TUAB (subset of TUEG)
     - ``pathology``
     - Lopez2017
   * - TUEV (subset of TUEG)
     - ``clinical_event``
     - Harati2015

Known limitations
~~~~~~~~~~~~~~~~~

* **Dropout at inference** -- The pretrained checkpoint stores
  ``drop_prob=0.1``; the paper uses 0 during fine-tuning.  If needed, this can
  be overridden via ``kwargs: {drop_prob: 0.0}`` in the model config.
