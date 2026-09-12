Age regression
==============

| **Name**: age
| **Category**: Others
| **Dataset**: :py:class:`~neuralset.studies.Shirazi2024` (HBN)
| **Objective**: :bdg-success:`Regression`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg age

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/age/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

Brain age prediction is the task of estimating a person's age from their brain signals [Engemann2022]_. It is typically studied in the context of neurological and psychiatric disorders, where deviations from typical brain aging patterns can serve as biomarkers for disease diagnosis and progression monitoring.

Dataset Notes
~~~~~~~~~~~~~

* `Shirazi2024` (HBN) contains EEG recordings from 11 cohorts ("releases") containing different participants. Here, we leave one release out for testing.
* The dataset contains different tasks (resting-state, contrast change detection, etc.). Here, we only use the resting-state data for age prediction.

References
~~~~~~~~~~

.. [Engemann2022] Engemann, Denis A., et al. "A reusable benchmark of brain-age prediction from
    M/EEG resting-state signals." Neuroimage 262 (2022): 119521.
