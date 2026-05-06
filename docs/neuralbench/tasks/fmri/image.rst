Image decoding
==============

| **Name**: image
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Allen2022MassiveRaw` (NSD, BIDS/deepprep variant)
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

The fMRI image decoding task involves retrieving the visual stimulus presented to a subject from their BOLD fMRI response. We use the Natural Scenes Dataset (NSD) [Allen2022]_, a large-scale 7T fMRI dataset in which 8 subjects each viewed up to 10,000 distinct natural scene images over 30--40 scanning sessions (~73,000 image presentations total).

BOLD responses are extracted from a 7-second window starting 2 seconds after image onset to account for the hemodynamic delay. Surface-projected activations (fsaverage5 mesh) are used as model input. Target embeddings are DINOv2-giant representations of the presented images (1536-dimensional).

The train/test split follows the dataset-native partition: NSD reserves a set of 1,000 images seen by all 8 subjects as a shared test set.

Dataset Notes
~~~~~~~~~~~~~

* The NSD is a very large dataset (several TB). Downloading requires AWS credentials and agreeing to the NSD terms of use.
* Only the ``betas_fithrf_GLMdenoise_RR`` single-trial beta maps and the stimulus images are needed.

References
~~~~~~~~~~

.. [Allen2022] Allen, Emily J., et al. "A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence." Nature Neuroscience 25.1 (2022): 116-126.
