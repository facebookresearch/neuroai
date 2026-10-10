Lateralized readiness potential (LRP) classification
=====================================================

| **Name**: lrp
| **Category**: brain-computer interfacing
| **Dataset**: ``Kappenman2020Lrp`` (ErpCore2021_LRP)
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg lrp

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/lrp/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The lateralized readiness potential (LRP) classification task involves distinguishing left- vs. right-hand responses from EEG recordings. The LRP is an asymmetric motor preparation component that develops before a voluntary hand movement, reflecting the differential activation of contralateral motor cortex. We use the Kappenman2020Lrp dataset [Kappenman2020Lrp]_, part of the ERP CORE (Compendium of Open Resources and Experiments), which contains EEG data from 40 subjects performing a flanker task requiring left- or right-hand button presses.

References
~~~~~~~~~~

.. [Kappenman2020Lrp] Kappenman, E. S., et al. "ERP CORE: An open resource for human event-related potential research." NeuroImage 225 (2021): 117465.
