Motor imagery classification
============================

| **Name**: motor imagery
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Stieger2021Continuous`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

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

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg motor_imagery --dataset schalk2004bci

References
~~~~~~~~~~

.. [Stieger2021] Stieger, James R., Stephen A. Engel, and Bin He. "Continuous sensorimotor rhythm based brain computer interface learning in a large population." Scientific Data 8.1 (2021): 98.
