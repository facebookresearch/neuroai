Schizophrenia diagnosis classification
=======================================

| **Name**: schizophrenia diagnosis
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Albrecht2019`
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg schizophrenia_diagnosis

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/schizophrenia_diagnosis/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task consists in predicting whether subjects have received a schizophrenia diagnosis (PSZ) or are healthy controls, based on their EEG recordings during a modified Simon task [Albrecht2019]_.

References
~~~~~~~~~~

.. [Albrecht2019] Albrecht, Matthew A., et al. "Increased conflict-induced slowing, but no differences in conflict-induced positive or negative prediction error learning in patients with schizophrenia." Neuropsychologia 123 (2019): 131-140.
