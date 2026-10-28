Audiovisual stimulus classification (MNE sample dataset)
=========================================================

| **Name**: audiovisual_stimulus
| **Category**: sensory
| **Dataset**: :py:class:`~neuralset.studies.Mne2013SampleEeg`
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Stratified trial-level split (single subject)

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg audiovisual_stimulus

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/audiovisual_stimulus/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

This task involves classifying audiovisual stimuli from EEG recordings into four categories: left ear auditory stimulus, right ear auditory stimulus, left visual field stimulus, or right visual field stimulus. The task uses the MNE-Python sample dataset [MNESample]_, which is a small, publicly available dataset designed for testing and demonstration purposes. We use it here as an  example of a low data regime task.

.. note::

   This is a **single-subject dataset** intended primarily for sanity-checking and integration testing. Because only one subject is available, the train/validation/test split is performed at the trial level (``split_by: _index``).

References
~~~~~~~~~~

.. [MNESample] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen. MNE software for processing MEG and EEG data. NeuroImage, 86, 446-460, 2014. https://mne.tools/stable/documentation/datasets.html#sample
