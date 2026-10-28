Motor imagery classification
============================

| **Name**: motor imagery
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Stieger2021Continuous`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

.. admonition:: 🏆 EEG/EMG Foundation Challenge 2026
   :class: tip

   This page describes the NeuralBench benchmark version of the task.
   :doc:`Track 2 -- BCI decoding
   </neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci>`
   of the challenge uses its streamed version, ``_motor_imagery_stream``,
   which scores each recording one window at a time, in time order, on the
   competition's training release by default.
   **Participants should use** ``neuralbench eeg _motor_imagery_stream``; the
   Track 2 page has its commands, timings, and competition data notes.

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg motor_imagery

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/motor_imagery/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The motor imagery classification task involves identifying different types of imagined motor movements from EEG recordings. Here, we use the Stieger2021 dataset [Stieger2021]_, which contains 64-channel EEG from 62 healthy participants who used motor imagery to control a cursor with continuous online visual feedback. The task has the following four classes:

* Right hand
* Left hand
* Both hands
* Rest

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Barachant2012`` (AlexMI) -- 8 subjects, 3 classes
* ``Cho2017`` -- 52 subjects, 2 classes
* ``Dornhege2004`` (BNCI2003_004) -- 5 subjects, 2 classes
* ``Dreyer2023`` -- 87 subjects, 2 classes
* ``Faller2012`` (BNCI2015_001) -- 12 subjects, 2 classes
* ``GrosseWentrup2009`` -- 10 subjects, 2 classes
* ``Leeb2007`` (BNCI2014_004) -- 9 subjects, 2 classes
* ``Lee2019Mi`` -- 54 subjects, 2 classes
* ``Liu2024Imagery`` -- 50 subjects (stroke patients), 2 classes
* ``Schalk2004Bci`` (EEGMIDB) -- 109 subjects, 4 classes (left fist, right fist, both fists, both feet)
* ``Scherer2012`` (BNCI2014_002) -- 14 subjects, 2 classes
* ``Schwarz2020`` (BNCI2020_001) -- 45 subjects, 3 classes
* ``Shin2017A`` -- 29 subjects, 2 classes
* ``Tangermann2012`` (BNCI2014_001) -- 9 subjects, 4 classes
* ``Wei2022A`` (Beetl2021_A) -- 3 subjects, 4 classes
* ``Wei2022B`` (Beetl2021_B) -- 2 subjects, 4 classes
* ``Zhou2016`` -- 4 subjects, 3 classes

The training release of the EEG/EMG Foundation Challenge 2026 Track 2 corpus is
available as `NEMAR nm000290, version 1.0.0
<https://doi.org/10.82901/nemar.nm000290>`__: 112 runs from 14 sessions of 10
participants, three cued mental commands (motor imagery, mental subtraction, word
generation) with the Graz and BrainHero interfaces, 41 EEG channels at 500 Hz. The
license is CC-BY-4.0.

.. code-block:: bash

   neuralfetch download Dreyer2026Proteus --path /path/to/DATA_DIR
   neuralbench eeg motor_imagery --dataset dreyer2026proteus

It is also the default dataset of ``_motor_imagery_stream``, which streams each
session in recording order and adds the per-session balanced accuracy. This
variant uses the task's subject-disjoint split. It is not the official
cross-session split or the sealed evaluation.

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg motor_imagery --dataset schalk2004bci2000

References
~~~~~~~~~~

.. [Stieger2021] Stieger, James R., Stephen A. Engel, and Bin He. "Continuous sensorimotor rhythm based brain computer interface learning in a large population." Scientific Data 8.1 (2021): 98.
