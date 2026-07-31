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

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional EEG image-decoding datasets can also be used with this task:

* ``Xu2024Alljoined`` (Alljoined1) -- 8 participants, 64-channel EEG at 512 Hz, viewing static images from the Natural Scenes Dataset (NSD) [Xu2024Alljoined]_.
* ``Xu2025Alljoined`` (Alljoined-1.6M) -- 20 participants viewing static images in EEG [Xu2025Alljoined]_. Source: `Hugging Face <https://huggingface.co/datasets/Alljoined/Alljoined-1.6M>`_.
* ``Grootswagers2022Human`` (THINGS-EEG1) -- 50 participants, 64-channel EEG at 1000 Hz, viewing rapid serial visual presentation (RSVP) streams covering all 1,854 THINGS object concepts; uses the dataset's predefined train/test split [Grootswagers2022]_.

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg image --dataset xu2024alljoined

References
~~~~~~~~~~

.. [Benchetrit2023] Benchetrit, Yohann, Hubert Banville, and Jean-Rémi King. "Brain decoding: toward real-time reconstruction of visual perception." arXiv preprint arXiv:2310.19812 (2023).
.. [Gifford2022Large] Gifford, Alessandro T., et al. "A large and rich EEG dataset for modeling human visual object recognition." NeuroImage 264 (2022): 119754.
.. [Hebart2019] Hebart, Martin N., et al. "THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images." PloS one 14.10 (2019): e0223792.
.. [Xu2024Alljoined] Xu, Jonathan, et al. "Alljoined -- A dataset for EEG-to-Image decoding." arXiv preprint arXiv:2404.05553 (2024).
.. [Xu2025Alljoined] Xu, Jonathan, et al. "Alljoined-1.6M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces." arXiv preprint arXiv:2508.18571 (2025).
.. [Grootswagers2022] Grootswagers, Tijl, et al. "Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams." Scientific Data 9.1 (2022): 3.
