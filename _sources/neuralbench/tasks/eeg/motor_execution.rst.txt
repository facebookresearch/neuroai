Motor execution classification
==============================

| **Name**: motor execution
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Srisrisawang2024Simultaneous` (BNCI2025_001)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg motor_execution

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/motor_execution/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The motor execution classification task involves identifying different types of executed motor movements from EEG recordings. Here, we use the Srisrisawang2024 dataset [Srisrisawang2024]_, which contains 60-channel EEG from 20 healthy participants performing discrete reaching movements. The 16 classes encode the joint combination of movement direction (up, down, left, right), speed (slow, fast), and distance (near, far).

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Ofner2017`` -- 15 subjects, 7 classes (executed upper limb movements: elbow, forearm, hand)
* ``Schirrmeister2017`` (HGD) -- 14 subjects, 4 classes (left hand, right hand, both feet, rest)

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg motor_execution --dataset schirrmeister2017

References
~~~~~~~~~~

.. [Srisrisawang2024] Srisrisawang, Nitikorn, and Gernot R. Müller-Putz. "Simultaneous encoding of speed, distance, and direction in discrete reaching: an EEG study." Journal of Neural Engineering 21.6 (2024): 066042.
