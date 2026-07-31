Reaction time prediction
========================

| **Name**: reaction time
| **Category**: others
| **Dataset**: :py:class:`~neuralset.studies.Shirazi2024` (HBN)
| **Objective**: :bdg-success:`Regression`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg reaction_time

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/reaction_time/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task corresponds to Challenge 1 ("Cross-Task Transfer Learning") of the `EEG Challenge 2025 <https://eeg2025.github.io/>`__ [Aristimunha2025]_.
The goal is to predict where, within a 2-s EEG window, participants responded to a visual stimulus with a mouse click.

The official challenge metric is the normalized root-mean-squared error (RMSE divided by the standard deviation of the ground-truth reaction times), reported as ``normalized_rmse`` alongside the other regression metrics.

Dataset Notes
~~~~~~~~~~~~~

* `Shirazi2024` (HBN) contains EEG recordings from 11 cohorts ("releases") containing different participants. Here, we leave one release out for testing.
* The dataset contains different tasks (resting-state, contrast change detection, etc.). Here, we use the contrast change detection (CCD) data as in [Aristimunha2025]_.

References
~~~~~~~~~~

.. [Aristimunha2025] Aristimunha, Bruno, et al. "EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding." arXiv preprint arXiv:2506.19141 (2025).
