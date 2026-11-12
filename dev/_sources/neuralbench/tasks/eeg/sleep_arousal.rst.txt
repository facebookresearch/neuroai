Sleep arousal detection
=======================

| **Name**: sleep_arousal
| **Category**: sleep
| **Dataset**: :py:class:`~neuralset.studies.Ghassemi2018` (Physionet Challenge 2018)
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg sleep_arousal

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sleep_arousal/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

Based on the PhysioNet/Computing in Cardiology Challenge 2018 [Ghassemi2018b]_, this
task benchmarks cortical-arousal detection from whole-night sleep EEG. Each
non-overlapping 15 s segment is labelled positive if it overlaps any scored
``SleepArousal`` event — respiratory disturbances (apnea, hypopnea, Cheyne-Stokes
breathing, hypoventilation, partial obstruction) or non-respiratory arousals
(RERA, spontaneous, PLM, bruxism, snore) — and negative otherwise.

This is intentionally a simpler framing than the original Challenge, which scored
per-sample, restricted positives to non-apnea arousals, and excluded wake and apnea
periods from scoring. We therefore do not expect scores to be directly comparable to
the Challenge leaderboard.

Dataset Notes
~~~~~~~~~~~~~

* Only the ``split == 'train'`` portion of the Challenge dataset (~994 subjects) is
  used, because the public test subset has no released labels. Train/val/test splits
  are drawn at the subject level (60/20/20).
* Most positive segments overlap respiratory events (apneas, hypopneas, RERAs),
  because the PC18 cohort is weighted toward sleep-disordered breathing.
* ``SleepArousal`` events tagged ``noise`` are artifact spans (vocalizations,
  coughing, movement, external noise) rather than cortical arousals, so they are
  filtered out before labelling. Segments overlapping only such spans receive
  the negative label rather than triggering a positive.
* Strong imbalance toward "no arousal" is addressed with a weighted sampler, and we
  report both ``bal_acc`` (principal metric, for consistency across the benchmark)
  and AUPRC.

References
~~~~~~~~~~

.. [Ghassemi2018b] Ghassemi, Mohammad M., et al. "You Snooze, You Win: the Physionet/Computing in Cardiology Challenge 2018." 2018 Computing in Cardiology Conference (CinC). Vol. 45. IEEE, 2018.
