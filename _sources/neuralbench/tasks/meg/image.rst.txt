Image decoding
==============

| **Name**: image
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Hebart2023ThingsMeg` (THINGS-MEG)
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench meg image

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/meg/image/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The image decoding task involves decoding visual stimuli from MEG recordings [Benchetrit2023b]_. In this task, we use the THINGS-MEG dataset [Hebart2023]_, which contains EEG data recorded while subjects viewed images from the THINGS database, a large-scale collection of naturalistic object images [Hebart2019b]_. The goal is to retrieve the presented image based on the MEG signals and a fixed pretrained image feature extractor.

References
~~~~~~~~~~

.. [Benchetrit2023b] Benchetrit, Yohann, Hubert Banville, and Jean-Rémi King. "Brain decoding: toward real-time reconstruction of visual perception." arXiv preprint arXiv:2310.19812 (2023).
.. [Hebart2023] Hebart, Martin N., et al. "THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior." Elife 12 (2023): e82580.
.. [Hebart2019b] Hebart, Martin N., et al. "THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images." PloS one 14.10 (2019): e0223792.
