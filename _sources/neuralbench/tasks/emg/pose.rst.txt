Hand pose decoding
==================

| **Name**: pose
| **Category**: motor / hand-pose decoding
| **Dataset**: :py:class:`~neuralset.studies.Salter2024Emg2pose` (emg2pose)
| **Objective**: :bdg-dark:`20-joint angle trajectory regression`
| **Split**: The paper's assignment, testing on its held-out user+stage set

.. image:: https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png
   :alt: emg2pose overview: sEMG wristband recordings paired with motion-capture hand pose
   :width: 75%
   :align: center

Usage
~~~~~

.. code-block:: bash

   # Download the NM000281 release
   neuralbench emg pose -m vemg2pose --download

   # Full paper configuration
   neuralbench emg pose -m vemg2pose

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml
      :language: yaml

Description
~~~~~~~~~~~

Hand-pose regression from 16-channel surface EMG against the 20 joint angles
of the UmeTrack hand skeleton [Salter2024]_: 25,253 recordings over 193
participants, 370 hours and 29 movement stages, with 2 kHz sEMG paired with
tracked joint angles. Each 5-s window is mapped to the 20-joint trajectory.

Joint angles stay in **radians**, the unit emg2pose trains and logs, so
``test/mae`` compares directly against its ``AngleMAE``. The paper's Table 4
reports that same quantity in degrees: multiply by 57.29578, which puts its
12.2-18.8 degrees at 0.213-0.328 radians.

This is the paper's **regression** setting (``regression_vemg2pose``), a plain
sequence-to-sequence map.  Its **tracking** setting is not implemented: that
one feeds in the initial pose and conditions on the previous state at each
step, which is a model-side change rather than a configuration one.

Dataset Notes
~~~~~~~~~~~~~

* **IK failures and padding**: BIDS events mark ``BAD_IK`` spans and bound the
  recording before the padded BDF tail. Those spans are blanked to ``NaN`` in
  the target channels, and any window overlapping one is dropped from every
  split -- emg2pose's ``skip_ik_failures``, rather than masking single frames.
* **Splits**: the paper's ``split`` and ``generalization`` labels are read from
  the session's BIDS ``scans.tsv``. NEMAR tags up to ``v1.0.3`` omit those two
  columns, so for those releases the labels are joined from the upstream
  ``emg2pose_metadata.csv`` on the ``scans.tsv`` ``source_file``.
* **Test scenario**: the paper splits ``test`` into three disjoint sets scored
  separately in its Table 4 -- held-out users, held-out stages, and both.  A
  single pooled score matches none of them, so ``test`` here keeps only the
  held-out user+stage set (456 recordings, 20 users, 7 h), which the paper calls
  "of greatest value as the most encompassing real-world deployment setting" and
  where ``vemg2pose`` regression scores 15.8 +- 1.4 degrees.  ``val`` keeps both
  of its scenarios, matching the validation split emg2pose selects models on.
  Note the paper averages within each user before reporting mean and standard
  deviation across users, whereas ``test/mae`` pools frames.
* **Rotation augmentation**: training rotates the band by -1, 0 or +1 electrode
  (the paper's Appendix C.4), and never touches validation or test.  emg2pose
  redraws the offset for every window; braindecode's ``BandRotation`` draws one
  per batch, so a training step sees one rotation rather than 64.

.. warning::

   emg2pose is released under CC-BY-NC-SA-4.0, and the UmeTrack hand model
   used for forward kinematics under CC-BY-NC-4.0.  Both are
   **non-commercial**.

References
~~~~~~~~~~

.. [Salter2024] Salter, Sasha, et al. "emg2pose: A large and diverse benchmark
   for surface electromyographic hand pose estimation." *Advances in Neural
   Information Processing Systems* 37 (2024).  arXiv:2412.02725.
