Alzheimer's disease and dementia classification
===============================================

| **Name**: dementia diagnosis
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Miltiadous2023`
| **Objective**: :bdg-secondary:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg dementia_diagnosis

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/dementia_diagnosis/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task consists in predicting whether subjects have received an Alzheimer's disease or frontotemporal dementia diagnosis, or are healthy, based on their eyes-closed resting-state EEG recordings [Miltiadous2023]_.

References
~~~~~~~~~~

.. [Miltiadous2023] Miltiadous, Andreas, et al. "A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG." Data 8.6 (2023): 95.
