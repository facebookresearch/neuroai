Pathology detection
===================

| **Name**: pathology
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Lopez2017` (TUAB)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg pathology

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/pathology/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The goal of the pathology detection task is to predict whether an EEG recording contains pathological EEG data or not [Lopez2017]_.

Dataset Notes
~~~~~~~~~~~~~

* Previous work (BIOT, LaBraM, CBraMod, etc.) evaluates test performance at the window level, rather than at the recording level like domain-specific SOTA papers (e.g. [Gemein2020]_). Here, we evaluate performance at the recording level.

References
~~~~~~~~~~

.. [Lopez2017] Lopez, Sebas, et al. "Automated identification of abnormal adult EEGs." 2015 IEEE signal processing in medicine and biology symposium (SPMB). IEEE, 2015.
.. [Gemein2020] Gemein, Lukas AW, et al. "Machine-learning-based diagnostics of EEG pathology." NeuroImage 220 (2020): 117021.
