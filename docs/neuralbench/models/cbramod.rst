CBraMod
=======

| **Config name**: ``CBraMod``
| **Pretrained checkpoint**: ``braindecode/cbramod-pretrained`` (HuggingFace Hub)
| **Reference**: Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*, 2025

CBraMod is a criss-cross transformer for EEG that decomposes attention into
separate spatial (S-Attention across channels) and temporal (T-Attention across
patches) streams.  The pretrained checkpoint was trained on the Temple
University Hospital EEG corpus (TUEG) at 200 Hz with a patch size of 200
samples (1 s) and 19 standard 10-20 channels. CBraMod accepts any number of
input channels without a fixed channel set or channel projection.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   CBraMod was pretrained on the full TUEG corpus.  The TUEG corpus overlaps
   with the NeuralBench ``pathology`` task (TUH Abnormal EEG Corpus),
   ``artifact`` task (TUH EEG Artifact Corpus), and ``clinical_event`` task
   (TUH EEG Events Corpus), all of which are subsets of TUEG.  Results on
   these tasks may therefore be inflated.

Known limitations
~~~~~~~~~~~~~~~~~

* ``n_times`` must be exactly divisible by ``patch_size`` (200).  Tasks with
  non-integer-second durations at 200 Hz will fail unless the window duration
  is adjusted.
* The spectral projection path contains a hardcoded reshape that assumes
  ``patch_size=200``.  Using a different patch size would require changes to
  the braindecode implementation.
