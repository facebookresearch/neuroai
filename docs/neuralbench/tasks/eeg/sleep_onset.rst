Sleep onset prediction
======================

| **Name**: sleep_onset
| **Category**: sleep
| **Dataset**: :py:class:`~neuralset.studies.Kemp2000` (Sleep-EDF)
| **Objective**: :bdg-success:`Regression`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg sleep_onset

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sleep_onset/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

Given a continuous EEG recording leading up to sleep, predict at every point in
time how many seconds remain before the participant falls asleep. Models output
a single regression value per analysis window, capped at 600 s (10 minutes
pre-onset). Inference is strictly causal: predictions for window ``[t - 5, t]``
may only depend on EEG up to time ``t``. The benchmark uses non-overlapping
5-s analysis windows -- one model input per 5-s slice of EEG -- so every
recording yields one prediction every 5 s of pre-onset signal.

Sleep onset is defined as **N2 onset**: the timestamp of the first scored N2
epoch in the polysomnogram-aligned hypnogram. N2 was chosen (rather than the
looser N1 or "first non-wake" definition) because it is the most reliably
scored stage across raters, the most reproducible across recording nights, and
corresponds to the consolidated transition into sleep that is clinically
meaningful [Rechtschaffen1968]_. The ``AddSleepOnsetTargets`` transform emits
a single ``SleepOnsetMarker`` event per recording spanning the trainable
pre-onset region, carrying the absolute ``n2_onset`` timestamp. The segmenter
tiles this marker via ``stride`` to produce 5-s analysis segments, and the
``SleepOnsetTargetExtractor`` computes
``clip(n2_onset - segment.stop, 0, 600)`` at extraction time -- so the target
is always derived from the actual segment boundary.

Evaluation
~~~~~~~~~~

The headline metric is **bMAE** (binned MAE): the mean absolute error
computed inside each of four ground-truth time-to-onset bins, averaged across
bins with equal weight. The bin boundaries map onto distinct physiological
regimes that demand fundamentally different algorithmic capabilities.

Standard regression metrics (MAE, RMSE, Pearson r, R^2, normalized RMSE) are
also logged alongside the headline ``bmae``.

References
~~~~~~~~~~

.. [Kemp2000] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
.. [Rechtschaffen1968] A Rechtschaffen, AE Kales. A manual of standardized terminology, techniques and scoring systems for sleep stages of human subjects. Los Angeles, CA: UCLA Brain Information Service. Brain Research Institute 10 (1968).
