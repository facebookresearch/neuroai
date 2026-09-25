LaBraM
======

| **Config name**: ``Labram``
| **Pretrained checkpoint**: ``braindecode/labram-pretrained`` (HuggingFace Hub)
| **Reference**: Jiang et al., *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI*, ICLR 2024

LaBraM is a patch-based EEG transformer with learned spatial and temporal
embeddings.  The pretrained checkpoint was trained on 16 public EEG datasets
totalling ~2534 hours at 200 Hz with a patch size of 200 samples (1 s) and up
to 128 channels from the standard 10-20/10-10 system. Input channels are
matched case-insensitively against the 128-channel ``LABRAM_CHANNEL_ORDER``;
unmatched channels are dropped.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   LaBraM was pretrained on 16 public EEG datasets totalling ~2534 hours
   (Appendix D of the paper).  The pretraining corpus includes **PhysioNet
   Motor Imagery** (Schalk et al., 2004) and **TUAR** (TUH EEG Artifact
   Corpus), both of which are also NeuralBench evaluation datasets.  Results
   on the affected tasks may be inflated because the model has seen the same
   *recordings* (though not the same *labels*) during self-supervised
   pretraining.

   The paper explicitly states that TUAB and TUEV recordings were excluded
   from pretraining, so ``pathology`` and ``clinical_event`` are not affected.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Pretraining dataset
     - NeuralBench task
     - NeuralBench study
   * - PhysioNet Motor Imagery
     - ``motor_imagery``
     - Schalk2009
   * - TUAR
     - ``artifact``
     - Hamid2020

Known limitations
~~~~~~~~~~~~~~~~~

* Zero-padding for non-divisible durations introduces artefactual silence at
  the end of the window, which may slightly reduce performance compared to
  natively aligned durations.
