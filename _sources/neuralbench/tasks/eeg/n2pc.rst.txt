N2pc attention classification
=============================

| **Name**: n2pc
| **Category**: visual neuroscience
| **Dataset**: ``Kappenman2020N2pc`` (ErpCore2021_N2pc)
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg n2pc

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/n2pc/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The N2pc classification task involves decoding the location of visual spatial attention from EEG recordings. The N2pc is an ERP component reflecting the deployment of visual-spatial attention to a target item in a visual search array, observable as a negative deflection contralateral to the attended item around 200-300 ms post-stimulus. We use the Kappenman2020N2pc dataset [Kappenman2020N2pc]_, part of the ERP CORE (Compendium of Open Resources and Experiments), which contains EEG data from 40 subjects performing a visual search task.


Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional dataset from MOABB can also be used with this task:

.. codespell:ignore-begin
* ``Reichert2020`` -- 18 subjects, covert spatial attention N2pc paradigm
.. codespell:ignore-end

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg n2pc --datasets reichert2020

References
~~~~~~~~~~

.. [Kappenman2020N2pc] Kappenman, E. S., et al. "ERP CORE: An open resource for human event-related potential research." NeuroImage 225 (2021): 117465.
