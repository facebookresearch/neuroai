Mental imagery classification
==============================

| **Name**: mental imagery
| **Category**: brain-computer interfacing
| **Dataset**: ``Scherer2015Individually`` (MOABB alias ``BNCI2015_004``)
| **Objective**: :bdg-info:`Multiclass classification` (5 classes)
| **Split**: PredefinedSplit, session 1 held out as test
| **Epoch window**: ``start = 3.0 s``, ``duration = 4.0 s`` (aligned with the imagery period of the paradigm)

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg mental_imagery

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/mental_imagery/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The mental imagery classification task decodes five distinct mental imagery tasks from 30-channel EEG recorded at 256 Hz while nine users with spinal-cord injury or stroke performed a cue-guided paradigm [Scherer2015]_. The classes, as they appear in the raw GDF annotations of the dataset, are **math** (mental arithmetic), **letter** (letter association / spelling), **rotation** (mental rotation), **count** (mental counting), and **baseline**. The dataset contains 3550 trials (710 per class, perfectly balanced) across 18 timelines (9 subjects x 2 sessions).

References
~~~~~~~~~~

.. [Scherer2015] Scherer, R., Faller, J., Friedrich, E. V. C., Opisso, E., Costa, U., Kuebler, A., and Mueller-Putz, G. R. "Individually Adapted Imagery Improves Brain-Computer Interface Performance in End-Users with Disability." PLOS ONE 10.5 (2015): e0123727. https://doi.org/10.1371/journal.pone.0123727
