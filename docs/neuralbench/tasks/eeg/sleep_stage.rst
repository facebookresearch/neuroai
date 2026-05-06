Sleep stage classification
==========================

| **Name**: sleep stage
| **Category**: sleep
| **Dataset**: :py:class:`~neuralset.studies.Kemp2000` (SleepEDFx)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg sleep_stage

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sleep_stage/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The sleep stage classification task involves categorizing segments of EEG data into different sleep stages based on established sleep scoring criteria [Rechtschaffen1968]_. The dataset used for this task is the Sleep-EDF dataset, which contains whole-night polysomnographic recordings from healthy subjects [Kemp2000]_.

Only sleep stage labels N1, N2, N3, REM, and Wake are considered for this task. Other labels such as Movement time and Unknown are excluded.

Dataset Notes
~~~~~~~~~~~~~

* Recordings are cropped to start 30 minutes before the first occurrence of a sleep stage and end 30 minutes after the last occurrence of a sleep stage to avoid imbalances due to long periods of wakefulness at the beginning and end of the recordings.

References
~~~~~~~~~~

.. [Rechtschaffen1968] A Rechtschaffen, AE Kales. A manual of standardized terminology, techniques and scoring systems for sleep stages of human subjects. Los Angeles, CA: UCLA Brain Information Service. Brain Research Institute 10 (1968).
.. [Kemp2000] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
