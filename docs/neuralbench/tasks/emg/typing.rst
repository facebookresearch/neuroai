Typing decoding
===============

| **Name**: typing
| **Category**: motor / input decoding
| **Dataset**: NM000104 (CTRL-Labs / Meta, [Sivakumar2024]_)
| **Objective**: :bdg-dark:`CTC sequence decoding`
| **Split**: Leave-subjects-out (cross-subject)

Usage
~~~~~

.. code-block:: bash

   # Auto-fetch NM000104 (~239 GB) via eegdash
   neuralbench emg typing -m emg2qwerty --download

   # Local 2-epoch sanity check
   neuralbench emg typing -m emg2qwerty --debug

   # Full benchmark run
   neuralbench emg typing -m emg2qwerty

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/typing/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Continuous-keystroke decoding from 32-channel surface EMG (two 16-electrode
wristbands at 2 kHz) using the CTC framework from [Sivakumar2024]_.  Each
5-s window (0.9 s + 4 s core + 0.1 s) is mapped to a variable-length
keystroke sequence over the 98-key paper vocabulary plus a CTC blank ―
99 output classes.

As compared to the original paper, the NeuralBench default configuration restricts training to the first ~10 subjects
(``timeline_index < 120``) and uses a leave-subjects-out split for
cross-subject evaluation, keeping turn-around tractable.

Dataset Notes
~~~~~~~~~~~~~

* **Auto-fetch**: ``--download`` pulls NM000104 from NEMAR
  (``s3://nemar/nm000104``) via :py:class:`neuralfetch.download.Eegdash`
  ― 1136 files, ~239 GB, under
  ``<DATA_DIR>/Sivakumar2024Emg2qwerty/download/sub-*/…``.
  Users with an existing BIDS copy should symlink it into ``download/``.
* **BIDS-aware reader**: the Study reads via
  :py:func:`mne_bids.read_raw_bids` (``≥ 0.18``); channel types come
  from the BIDS sidecars.  On ``mne_bids ≥ 0.19`` the EMG-unit fix
  lands the data in microvolts directly; on older versions
  ``BidsEmg._read`` applies a V→µV rescale.
* **Windowing**: 5-s sliding windows with a 4-s stride (0.9 s left +
  4-s core + 0.1 s right); ``CroppedExtractor`` restricts label
  collection to the core, so the 4-s cores tile each session
  non-overlappingly while the EMG signal context overlaps neighbours
  by 1 s.  The paper [Sivakumar2024]_ trains on 4-s windows but feeds
  whole sessions at test time; we apply this 5-s padded window across
  all splits ― slightly pessimistic CER, tracked as a follow-up.

References
~~~~~~~~~~

.. [Sivakumar2024] Sivakumar, Viswanath, et al. "emg2qwerty: A large
   dataset with baselines for touch typing using surface electromyography."
   *Advances in Neural Information Processing Systems* 37 (2024): 91373--91389.
