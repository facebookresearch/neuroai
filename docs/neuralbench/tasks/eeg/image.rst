Image decoding
==============

| **Name**: image
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Gifford2022Large` (THINGS-EEG2)
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg image

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/image/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The image decoding task involves decoding visual stimuli from EEG recordings [Benchetrit2023]_. In this task, we use the Gifford2022Large dataset [Gifford2022Large]_, which contains EEG data recorded while subjects viewed images from the THINGS database, a large-scale collection of naturalistic object images [Hebart2019]_. The goal is to retrieve the presented image based on the EEG signals and a fixed pretrained image feature extractor.

Dataset Notes
~~~~~~~~~~~~~

* We use [Gifford2022Large]_ for evaluation because test images were recorded in separate runs, limiting the potential impact of temporal correlations on decoding performance.

References
~~~~~~~~~~

.. [Benchetrit2023] Benchetrit, Yohann, Hubert Banville, and Jean-Rémi King. "Brain decoding: toward real-time reconstruction of visual perception." arXiv preprint arXiv:2310.19812 (2023).
.. [Gifford2022Large] Gifford, Alessandro T., et al. "A large and rich EEG dataset for modeling human visual object recognition." NeuroImage 264 (2022): 119754.
.. [Hebart2019] Hebart, Martin N., et al. "THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images." PloS one 14.10 (2019): e0223792.
