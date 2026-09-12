Psychopathology prediction
==========================

| **Name**: psychopathology
| **Category**: Phenotyping
| **Dataset**: :py:class:`~neuralset.studies.Shirazi2024` (HBN)
| **Objective**: :bdg-success:`Regression`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg psychopathology

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/psychopathology/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task corresponds to Challenge 2 ("Externalizing Factor Prediction / Subject-Invariant Representation") of the `EEG Challenge 2025 <https://eeg2025.github.io/>`__ [Aristimunha2025]_.
The goal is to predict the externalizing psychopathology factor -- a continuous score derived from the Child Behavior Checklist (CBCL) -- from 2-s resting-state EEG windows. This probes whether models can learn generalizable neural representations that transfer across subjects while recovering clinically relevant variability.

The official challenge metric is the normalized root-mean-squared error (RMSE divided by the standard deviation of the ground-truth externalizing scores), reported as ``normalized_rmse`` alongside the other regression metrics.

Dataset Notes
~~~~~~~~~~~~~

* `Shirazi2024` (HBN) contains EEG recordings from 11 cohorts ("releases") containing different participants. Here, we leave one release out for testing.
* The dataset contains different tasks (resting-state, contrast change detection, etc.). Here, we only use the resting-state data for psychopathology prediction.
* Only subjects with a non-null externalizing CBCL score are retained.

References
~~~~~~~~~~

.. [Aristimunha2025] Aristimunha, Bruno, et al. "EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding." arXiv preprint arXiv:2506.19141 (2025).
