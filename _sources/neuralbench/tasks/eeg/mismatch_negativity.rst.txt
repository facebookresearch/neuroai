Mismatch negativity (MMN) classification
=========================================

| **Name**: mismatch negativity
| **Category**: auditory neuroscience
| **Dataset**: ``Kappenman2020Mmn`` (ErpCore2021_MMN)
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg mismatch_negativity

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/mismatch_negativity/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The mismatch negativity (MMN) classification task involves classifying EEG epochs as responses to standard vs. deviant auditory stimuli. The MMN is an automatic brain response to unexpected changes in an otherwise regular auditory sequence, and is widely used as a marker of pre-attentive auditory change detection. We use the Kappenman2020Mmn dataset [Kappenman2020Mmn]_, part of the ERP CORE (Compendium of Open Resources and Experiments), which contains EEG data from 40 subjects passively listening to tone sequences with occasional deviant tones.

References
~~~~~~~~~~

.. [Kappenman2020Mmn] Kappenman, E. S., et al. "ERP CORE: An open resource for human event-related potential research." NeuroImage 225 (2021): 117465.
