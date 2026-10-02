Mental Workload Classification
===============================

| **Name**: mental_workload
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Hinss2022`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg mental_workload

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/mental_workload/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task involves classifying mental workload levels from EEG recordings. The Hinss2022 dataset [Hinss2022]_ contains recordings from 29 participants performing 4 cognitive tasks designed to elicit distinct mental states: Psychomotor Vigilance Task (PVT), Flanker task, N-back tasks (zero-back, one-back, two-back), and Multi-Attribute Task Battery at different difficulty levels (easy, medium, difficult).

Here, we focus on the MATB task data, and use the different MATB levels (easy, medium, difficult) as the target labels to represent varying degrees of working memory load.

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Hinss2021`` -- 15 subjects, 4 classes (resting state, easy, medium, difficult)
* ``Jao2021`` (BNCI2022_001) -- 13 subjects, drone piloting at varying difficulty levels

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg mental_workload --datasets hinss2021

References
~~~~~~~~~~

.. [Hinss2022] Hinss, Marcel F., et al. "Open multi-session and multi-task EEG cognitive Dataset for passive brain-computer Interface Applications." Scientific Data 10.1 (2023): 85.
