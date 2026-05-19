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

Continuous-keystroke decoding from 32-channel surface EMG (two 16-electrode
wristbands at 2 kHz) using the CTC framework from [Sivakumar2024]_.  Each
5-s window (0.9 s + 4 s core + 0.1 s) is mapped to a variable-length
keystroke sequence over the 98-key paper vocabulary plus a CTC blank ―
99 output classes.

The 51-class compact variant case-folds letters, US-QWERTY-folds shifted
symbols (``!`` → ``1``, ``~`` → `` ` ``, …), and drops ``Key.shift`` ―
useful when a downstream LM can recover the missing shift state.

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch**: ``--download`` pulls NM000104 from NEMAR
  (``s3://nemar/nm000104``) via :py:class:`neuralfetch.download.Eegdash`
  ― 1136 files, ~239 GB, under ``<DATA_DIR>/Emg2qwerty/download/sub-*/…``.
  An existing BIDS tree placed directly under ``<DATA_DIR>/Emg2qwerty/``
  is also picked up.
* **BIDS-aware reader**: the Study reads via
  :py:func:`mne_bids.read_raw_bids` (``≥ 0.19``); channel types and EMG
  units come from the BIDS sidecars, no manual coercion or rescaling.
* **CTC on MPS**: ``nn.CTCLoss`` isn't implemented on Apple MPS.
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` keeps the rest of the model on MPS
  while CTC falls back to CPU; paper-level CER still requires a GPU.
* **Test-time windowing**: the paper trains on 4-s windows but feeds
  whole sessions at test time.  We apply the same 4-s window across all
  splits ― slightly pessimistic CER; tracked as a follow-up.

References
~~~~~~~~~~

.. [Sivakumar2024] Sivakumar et al., "emg2qwerty: A Large Dataset with
   Baselines for Touch Typing using Surface Electromyography."
   *Advances in Neural Information Processing Systems* 37, 91373--91389, 2024.
