Qwerty keystroke decoding
=========================

| **Name**: qwerty
| **Category**: motor / input decoding
| **Dataset**: :py:class:`~neuralbench.tasks.emg.qwerty.study.Emg2qwerty` (NM000104)
| **Objective**: :bdg-dark:`CTC sequence decoding`
| **Split**: Leave-sessions-out (personalized) or leave-subjects-out (generic)

Usage
~~~~~

.. code-block:: bash

   neuralbench emg qwerty -m emg2qwerty --download            # 1. fetch NM000104 (~239 GB) via eegdash
   neuralbench emg qwerty -m emg2qwerty --debug               # 2. local 2-epoch sanity check
   neuralbench emg qwerty -m emg2qwerty --dataset paper_personalized
   neuralbench emg qwerty -m emg2qwerty --dataset paper_generic

   # Fine-tune from the upstream emg2qwerty generic checkpoint:
   neuralbench emg qwerty -m emg2qwerty \
       --dataset paper_personalized_finetune \
       --checkpoint /path/to/upstream_generic.ckpt

   # Compact 51-class vocabulary (case + US-QWERTY shift folded):
   neuralbench emg qwerty -m emg2qwerty --dataset paper_personalized_compact

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/qwerty/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Continuous-keystroke decoding from 32-channel surface EMG (two 16-electrode
wristbands at 2 kHz) using the CTC framework introduced in [Sivakumar2024]_.
Each 5-second EMG window (0.9 s left context + 4 s core + 0.1 s right context)
is mapped to a variable-length keystroke sequence over a 98-key vocabulary
(letters + digits + punctuation + 4 modifiers) plus a CTC blank, for a total
of 99 output classes.

Four reproducible recipes ship as ``--dataset`` overlays:

* ``paper_personalized`` — train on N-1 sessions of one subject, validate on
  the held-out session (paper headline ≈ 10 % CER after 30-epoch fine-tune
  on top of a generic-pretrained backbone).
* ``paper_generic`` — train on N-1 subjects, validate on the held-out one
  (paper headline ≈ 30 % CER at 150 epochs).
* ``paper_personalized_finetune`` — pair with ``--checkpoint
  /path/to/upstream_generic.ckpt``.  Lowers ``max_lr`` 100× and defers
  augmentation start so the from-scratch recipe doesn't overshoot the
  pretrained minimum.  Upstream's ``model.4.{weight,bias}`` rename to
  ``final_layer.{weight,bias}`` is handled automatically by
  ``EMG2QwertyNet.mapping`` inside :func:`neuralbench.utils.load_checkpoint`.
* ``paper_personalized_compact`` — same data subset as
  ``paper_personalized`` but with the compact 51-class vocabulary
  (case + US-QWERTY shift folded; ``Key.shift`` dropped).  Roughly
  halves the output space.  Train from scratch (not compatible with
  ``--checkpoint`` against upstream's 99-class head).

The from-scratch overlays only carry the data-subset deltas (subject
filter + split mode); the optimizer (Adam + OneCycleLR, lr 1e-8 →
1e-3 → 1e-6 with 40 %
warmup) lives in the task ``config.yaml`` and the SpecAugment / band-rotation
augmentation callbacks live in ``models/emg2qwerty.yaml`` (they bind to the
``EMG2QwertyNet`` spectrogram and the 2 × 16 wristband layout).

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch via eegdash**: ``--download`` pulls NM000104 from NEMAR
  (``s3://nemar/nm000104``) using
  :py:class:`neuralfetch.download.Eegdash` -- 1136 files, ~239 GB.  Files
  land at ``<DATA_DIR>/Emg2qwerty/download/sub-XXXXXXXX/...``.  If you
  already have a BIDS tree placed directly under
  ``<DATA_DIR>/Emg2qwerty/``, the study source picks that up too (no
  re-download).
* **µV rescale**: the upstream emg2qwerty HDF5 stores EMG in microvolts
  while ``mne.io.read_raw_bdf`` returns volts. ``Emg2qwertyRaw`` multiplies
  by 1e6 on read so the spectrogram log-floor doesn't cap gradients.
* **CTC + MPS**: ``nn.CTCLoss`` is not implemented on Apple MPS; the pipeline
  sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so the rest of the model runs on
  MPS while CTC falls back to CPU. Reaching paper-level CER requires a
  proper GPU.
* **Smoke runs collapse to all-blank**: short / small-data runs converge
  to the trivial all-blank CTC minimum within an epoch. This is expected
  and matches the upstream reference at the same scale.

References
~~~~~~~~~~

.. [Sivakumar2024] Sivakumar, V., Seely, J., Du, A., Bittner, S. R.,
   Berenzweig, A., Bolarinwa, A., Gramfort, A., & Mandel, M. I. (2024).
   "emg2qwerty: A Large Dataset with Baselines for Touch Typing using
   Surface Electromyography." *Advances in Neural Information Processing
   Systems* 37, 91373--91389.
