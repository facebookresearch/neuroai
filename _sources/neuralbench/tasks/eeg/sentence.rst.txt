Sentence decoding
=================

| **Name**: sentence
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Hollenstein2018` (ZuCo)
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg sentence

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sentence/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The sentence decoding task involves decoding sentence stimuli from EEG segments. In this task, we use the ZuCo dataset [Hollenstein2018]_, which contains EEG data recorded while subjects read sentences presented one at a time on a screen.

Dataset Notes
~~~~~~~~~~~~~

* To avoid sentence leakage, the SR task recordings are used for testing, while the NR and TSR task recordings are concatenated and used for training and validation.
* To ensure a large enough number of examples for training, sliding window segments are extracted within each sentence presentation, and are mapped to the same sentence representation.

References
~~~~~~~~~~

.. [Hollenstein2018] Hollenstein, Nora, et al. "ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading." Scientific data 5.1 (2018): 1-13.
