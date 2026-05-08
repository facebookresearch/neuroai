Qwerty keystroke decoding
=========================

| **Name**: qwerty
| **Category**: motor / input decoding
| **Dataset**: NM000104 (CTRL-Labs / Meta, [Sivakumar2024]_)
| **Objective**: :bdg-dark:`CTC sequence decoding`
| **Split**: Leave-sessions-out (personalized) or leave-subjects-out (generic)

Usage
~~~~~

.. code-block:: bash

   # Auto-fetch NM000104 (~239 GB) via eegdash
   neuralbench emg qwerty -m emg2qwerty --download

   # Local 2-epoch sanity check
   neuralbench emg qwerty -m emg2qwerty --debug

   # Personalized (N-1 sessions of one subject)
   neuralbench emg qwerty -m emg2qwerty --dataset paper_personalized

   # Generic cross-subject baseline
   neuralbench emg qwerty -m emg2qwerty --dataset paper_generic

   # Compact 51-class vocabulary (case + US-QWERTY shift folded)
   neuralbench emg qwerty -m emg2qwerty --dataset paper_personalized_compact

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/qwerty/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Continuous-keystroke decoding from 32-channel surface EMG (two
16-electrode wristbands at 2 kHz) using the CTC framework introduced
in [Sivakumar2024]_.  Each 5-second EMG window (0.9 s left context +
4 s core + 0.1 s right context) is mapped to a variable-length
keystroke sequence over the paper's 98-key vocabulary (letters +
digits + punctuation + 4 modifiers) plus a CTC blank, for 99 output
classes total.

A compact 51-class variant collapses uppercase letters with their
lowercase counterparts, US-QWERTY shifted symbols with their unshifted
forms (``!`` → ``1``, ``~`` → `` ` ``, etc.), and drops the
``Key.shift`` modifier — useful when the downstream task can recover
case + punctuation from a language model.

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch via eegdash**: ``--download`` pulls NM000104 from
  NEMAR (``s3://nemar/nm000104``) using
  :py:class:`neuralfetch.download.Eegdash` — 1136 files, ~239 GB,
  landing under ``<DATA_DIR>/Emg2qwerty/download/sub-*/...``.  An
  existing BIDS tree placed directly under ``<DATA_DIR>/Emg2qwerty/``
  is also picked up.
* **µV rescale**: upstream's HDF5 stores EMG in microvolts while
  ``mne.io.read_raw_bdf`` returns volts; ``Emg2qwertyRaw`` multiplies
  by 1e6 on read so the spectrogram log-floor doesn't cap gradients.
* **CTC + MPS**: ``nn.CTCLoss`` isn't implemented on Apple MPS;
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` keeps the rest of the model on
  MPS while CTC falls back to CPU.  Paper-level CER requires a real
  GPU.
* **Test-time windowing gap**: the upstream paper trains on fixed
  4-s windows but feeds whole sessions at test time.  Our pipeline's
  segmenter applies the same 4-s window across all splits — slightly
  pessimistic for CER reporting; tracked as a follow-up.

References
~~~~~~~~~~

.. [Sivakumar2024] Sivakumar, V., Seely, J., Du, A., Bittner, S. R.,
   Berenzweig, A., Bolarinwa, A., Gramfort, A., & Mandel, M. I. (2024).
   "emg2qwerty: A Large Dataset with Baselines for Touch Typing using
   Surface Electromyography." *Advances in Neural Information Processing
   Systems* 37, 91373--91389.
