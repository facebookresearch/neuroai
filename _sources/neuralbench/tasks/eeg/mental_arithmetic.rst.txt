Mental arithmetic detection
===========================

| **Name**: mental arithmetic
| **Category**: mental state classification
| **Dataset**: :py:class:`~neuralset.studies.Zyma2018`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg mental_arithmetic

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/mental_arithmetic/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task involves classifying EEG signals to detect whether the participants were performing a mental arithmetic task or not [Zyma2018]_.

Dataset Notes
~~~~~~~~~~~~~

* Previous work (CBraMod, EEG-FM-Bench) framed this binary classification problem as a "mental stress" detection problem. However, as the non-arithmetic condition did not control for the cognitive workload induced by the task, we posit that the most obvious signal models can rely on for the classification task will be related to the correlates of mental arithmetic.

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional dataset from MOABB can also be used with this task:

* ``Shin2017B`` -- 29 subjects, 2 classes (serial subtraction vs resting baseline)

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg mental_arithmetic --datasets shin2017b

References
~~~~~~~~~~

.. [Zyma2018] Zyma, Igor, et al. "Electroencephalograms during mental arithmetic task performance." Data 4.1 (2019): 14.
