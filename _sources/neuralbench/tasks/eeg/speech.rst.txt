Speech decoding
===============

| **Name**: speech
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Brennan2019`
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Leave-last-runs-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg speech

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/speech/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The speech decoding task involves decoding speech stimuli from EEG recordings [Defossez2023b]_. In this task, we use the Brennan2019 dataset [Brennan2019]_, which contains EEG data recorded while listened to an audiobook for ~12 minutes.

Dataset Notes
~~~~~~~~~~~~~

* In [Brennan2019]_, participants listened to 12 minutes of an audiobook. We use the last 20% of sentences played during the recordings for testing, and the remaining runs for training and validation.

References
~~~~~~~~~~

.. [Defossez2023b] Défossez, Alexandre, et al. "Decoding speech perception from non-invasive brain recordings." Nature Machine Intelligence 5.10 (2023): 1097-1107.
.. [Brennan2019] Brennan, Jonathan R., and John T. Hale. "Hierarchical structure guides rapid linguistic predictions during naturalistic listening." PloS one 14.1 (2019): e0207741.
