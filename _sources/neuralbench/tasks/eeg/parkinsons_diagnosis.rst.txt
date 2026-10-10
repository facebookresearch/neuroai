Parkinson's disease diagnosis classification
============================================

| **Name**: parkinsons diagnosis
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Singh2021`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg parkinsons_diagnosis

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/parkinsons_diagnosis/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task consists in predicting whether a subject has received a Parkinson's disease (PD) diagnosis or is a healthy control (HC) based on EEG recordings during an interval timing task, where participants had to press a key after estimating a time interval of either 3 or 7 seconds.

Dataset Notes
~~~~~~~~~~~~~

* We use the re-packaged version of Singh2021 by [Kastrati2025]_, available on HuggingFace. Sessions with insufficient data or poor signal quality, or with fewer than 20 valid key presses per interval condition were excluded from analyses [Singh2021]_.

References
~~~~~~~~~~

.. [Singh2021] Singh, Arun, et al. "Timing variability and midfrontal~ 4 Hz rhythms correlate with cognition in Parkinson’s disease." npj Parkinson's Disease 7.1 (2021): 14.
.. [Kastrati2025] Kastrati, Ard, et al. "EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications." arXiv preprint arXiv:2512.08959 (2025).
