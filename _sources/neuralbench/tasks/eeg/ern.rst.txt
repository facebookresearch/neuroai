Error related-negativity (ERN) classification
=============================================

| **Name**: error-related negativity
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Kappenman2021ErpErn`
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg ern

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/ern/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The error related-negativity (ERN) classification task involves identifying error-related potentials in EEG recordings. ERNs are brain responses elicited when a person makes a mistake, often used in BCI applications. In this task, we use the Kappenman2021ErpErn dataset [Kappenman2021]_, part of the ERP CORE (Compendium of Open Resources and Experiments), which contains EEG data from 40 subjects performing a flanker task that elicits ERN responses on error trials.

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Chavarriaga2010`` (BNCI2015_013) -- 6 subjects, observation-based ERN from a moving cursor paradigm
* ``Kueper2024`` (IntEr-HRI) -- 8 subjects, error-related potentials from orthosis-induced movement errors

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg ern --datasets kueper2024

References
~~~~~~~~~~

.. [Kappenman2021] Kappenman, E. S., et al. "ERP CORE: An open resource for human event-related potential research." NeuroImage 225 (2021): 117465.
