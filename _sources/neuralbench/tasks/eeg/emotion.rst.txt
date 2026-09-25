Emotion classification
======================

| **Name**: emotion
| **Category**: mental state classification
| **Dataset**: :py:class:`~neuralset.studies.Chen2023` (FACED)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg emotion

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/emotion/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task involves classifying emotional states based on EEG signals. The dataset used for this task is the FACED dataset, which contains EEG recordings from subjects exposed to videos chosen to elicit specific emotions [Chen2023]_. The goal is to categorize the EEG segments into one of 9 different emotional states: anger, fear, disgust, sadness, neutral, amusement, tenderness, inspiration, and joy.

Dataset Notes
~~~~~~~~~~~~~

* As in [Ye2024]_, [Wang2025]_ and [Ding2025]_, we perform a leave-subjects-out split of the subjects into train, validation, and test splits. Since each subject saw the same video stimuli, this means the models can learn to recognize specific video stimuli from the EEG data, rather than actually predict the corresponding emotion label.

References
~~~~~~~~~~

.. [Chen2023] Chen, Jingjing, et al. "A large finer-grained affective computing EEG dataset." Scientific Data 10.1 (2023): 740.
.. [Wang2025] Wang, Jiquan, et al. "Cbramod: A criss-cross brain foundation model for eeg decoding." arXiv preprint arXiv:2412.07236 (2024).
.. [Ye2024] Ye, Weishan, et al. "Semi-supervised dual-stream self-attentive adversarial graph contrastive learning for cross-subject eeg-based emotion recognition." IEEE Transactions on Affective Computing (2024).
.. [Ding2025] Ding, Yi, et al. "EmT: A novel transformer for generalized cross-subject EEG emotion recognition." IEEE Transactions on Neural Networks and Learning Systems (2025).
