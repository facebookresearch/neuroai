Depression diagnosis classification
===================================

| **Name**: depression diagnosis
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Mumtaz2018`
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg depression_diagnosis

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/depression_diagnosis/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task consists in predicting whether subjects have received a major depressive disorder (MDD) diagnosis, or are healthy, based on their EEG recordings in eyes closed, eyes open, and P300 task conditions [Mumtaz2018]_.

References
~~~~~~~~~~

.. [Mumtaz2018] Mumtaz, Wajid, et al. "A machine learning framework involving EEG-based functional connectivity to diagnose major depressive disorder (MDD)." Medical & biological engineering & computing 56 (2018): 233-246.
