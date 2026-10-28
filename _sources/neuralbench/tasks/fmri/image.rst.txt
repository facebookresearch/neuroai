Image decoding
==============

| **Name**: image
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralfetch.studies.chang2019bold5000.Chang2019Bold5000` (BOLD5000)
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench fmri image

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/fmri/image/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The fMRI image decoding task involves retrieving the visual stimulus presented to a subject from their BOLD fMRI response. We use BOLD5000 [Chang2019]_, a 3T fMRI dataset in which 4 subjects each viewed ~5,000 natural images drawn from the SUN, COCO, and ImageNet collections over 9--15 scanning sessions.

BOLD responses are extracted from a 7-second window starting 2 seconds after image onset to account for the hemodynamic delay. Surface-projected activations (fsaverage5 mesh) are used as model input. Target embeddings are DINOv2-giant representations of the presented images (1536-dimensional).

The train/test split follows the dataset-native partition based on the 113 images repeated across sessions for test-retest reliability.

Dataset Notes
~~~~~~~~~~~~~

* BOLD5000 is distributed via OpenNeuro (``ds001499``) under a CC0 licence; the preprocessed beta maps (MNI152NLin2009aSym space) and the stimulus images are needed.

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional fMRI image-decoding datasets can also be used with this task:

* ``Allen2022MassiveRaw`` (NSD) -- 8 participants each viewing up to 10,000 distinct natural scene images over 30--40 7T scanning sessions (~73,000 presentations total) [Allen2022]_.
* ``Hebart2023ThingsBold`` (THINGS-fMRI) -- 3 participants viewing 1,854 diverse object images from the THINGS database [Hebart2023]_.

To run with an alternate dataset:

.. code-block:: bash

   neuralbench fmri image --dataset Allen2022MassiveRaw

References
~~~~~~~~~~

.. [Allen2022] Allen, Emily J., et al. "A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence." Nature Neuroscience 25.1 (2022): 116-126.
.. [Chang2019] Chang, Nadine, et al. "BOLD5000, a public fMRI dataset while viewing 5000 visual images." Scientific Data 6.1 (2019): 49.
.. [Hebart2023] Hebart, Martin N., et al. "THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior." eLife 12 (2023): e82580.
