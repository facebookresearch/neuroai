Seizure detection
=================

| **Name**: seizure
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Dan2023` (CHB-MIT)
| **Objective**: :bdg-info:`Multiclass classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg seizure

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/seizure/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The seizure detection task involves identifying epileptic seizures from EEG recordings [Shoeb2009]_. The dataset used for this task is the CHB-MIT dataset, which contains EEG data from pediatric subjects with intractable seizures [Guttag2010]_. Here, we use the BIDS-compatible version of the dataset [Dan2023]_. We combine all seizure types into a single positive class for this binary classification task.

Dataset Notes
~~~~~~~~~~~~~

* Subject 21 in the original study [Guttag2010]_ was actually the second session of subject 1.

References
~~~~~~~~~~

.. [Dan2023] Dan, J.and A. Shoeb. BIDS CHB-MIT Scalp EEG Database. v1.0.0, EPFL, 5 Dec. 2023, https://doi.org/10.5281/zenodo.10259996.
.. [Guttag2010] Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2K01R
.. [Shoeb2009] Shoeb, Ali Hossam. Application of machine learning to epileptic seizure onset detection and treatment. Diss. Massachusetts Institute of Technology, 2009.
