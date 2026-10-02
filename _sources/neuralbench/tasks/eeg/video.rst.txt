Video decoding
==============

| **Name**: video
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Liu2024` (SEED-DV)
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Leave-concepts-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg video

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/video/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The video decoding task involves decoding visual stimuli from EEG recordings. In this task, we use the SEED-DV dataset [Liu2024]_, which contains EEG data recorded while subjects watched 1,400 2-s video clips representing 40 concepts. The goal is to retrieve the presented video based on the EEG signals and a fixed pretrained video feature extractor.

Dataset Notes
~~~~~~~~~~~~~

* The 40 concept labels used in this task were manually inferred from BLIP-generated captions and are not official labels from the original SEED-DV dataset. They should be treated as approximate semantic groupings of the video stimuli.
* The dataset must be manually downloaded after requesting access from the original authors.

References
~~~~~~~~~~

.. [Liu2024] Liu, Xuan-Hao, et al. "EEG2video: Towards decoding dynamic visual perception from EEG signals." Advances in Neural Information Processing Systems 37 (2024): 72245-72273.
