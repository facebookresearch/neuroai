Steady-state visually evoked potential (SSVEP) classification
=============================================================

| **Name**: ssvep
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Wang2017Benchmark` (Wang2016)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg ssvep

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/ssvep/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The visual steady-state evoked potential (SSVEP) classification task involves identifying different visual stimulus frequencies from EEG recordings. SSVEPs are brain responses elicited by visual stimuli flickering at specific frequencies. In this task, we use the Wang2017 benchmark dataset [Wang2017]_, which contains 64-channel EEG from 34 healthy participants fixating on targets in a 40-class SSVEP speller using joint frequency-phase modulation (8-15.8 Hz with 0.2 Hz spacing).

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Kalunga2016`` -- 12 subjects, 4 classes
* ``Lee2019Ssvep`` -- 54 subjects, 4 classes (5.45, 6.67, 8.57 and 12 Hz at four positions)
* ``Nakanishi2015`` -- 9 subjects, 12 classes
* ``Oikonomou2016A`` (MAMEM1) -- 11 subjects, 5 classes
* ``Oikonomou2016B`` (MAMEM2) -- 11 subjects, 5 classes
* ``Oikonomou2016C`` (MAMEM3) -- 11 subjects, 5 classes

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg ssvep --datasets lee2019ssvep

References
~~~~~~~~~~

.. [Wang2017] Wang, Yijun, Xiaogang Chen, Xiaorong Gao, and Shangkai Gao. "A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces." IEEE Transactions on Neural Systems and Rehabilitation Engineering 25.10 (2017): 1746-1752.
