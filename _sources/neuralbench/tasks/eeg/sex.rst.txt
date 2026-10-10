Sex regression
==============

| **Name**: sex
| **Category**: Others
| **Dataset**: :py:class:`~neuralset.studies.Shirazi2024` (HBN)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg sex

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sex/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

Brain sex prediction is the task of estimating a person's sex from their brain signals [Khayretdinova2025]_.

Dataset Notes
~~~~~~~~~~~~~

* `Shirazi2024` (HBN) contains EEG recordings from 11 cohorts ("releases") containing different participants. Here, we leave one release out for testing.
* The dataset contains different tasks (resting-state, contrast change detection, etc.). Here, we only use the resting-state data for age prediction.

References
~~~~~~~~~~

.. [Khayretdinova2025] Khayretdinova, Mariam, et al. "Prediction of brain sex from EEG: using large-scale heterogeneous dataset for developing a highly accurate and interpretable ML model." NeuroImage 285 (2024): 120495.
