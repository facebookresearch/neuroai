Code-modulated visually evoked potential (c-VEP) classification
================================================================

| **Name**: cvep
| **Category**: brain-computer interfacing
| **Dataset**: :py:class:`~neuralset.studies.Castillos2023Cvep100`
| **Objective**: :bdg-info:`Binary classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg cvep

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/cvep/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The code-modulated visually evoked potential (c-VEP) classification task involves identifying different visual stimulus codes from EEG recordings. c-VEPs are brain responses elicited by visual stimuli that are modulated using pseudo-random binary sequences, rather than periodic flickering at fixed frequencies as in SSVEP.

Additional Datasets
~~~~~~~~~~~~~~~~~~~

The following additional datasets from MOABB can also be used with this task:

* ``Castillos2023BurstVep100`` -- 12 subjects, burst-VEP at 100 Hz
* ``Castillos2023BurstVep40`` -- 12 subjects, burst-VEP at 40 Hz
* ``Castillos2023Cvep40`` -- 12 subjects, c-VEP at 40 Hz
* ``MartinezCagigal2023Checker`` -- 16 subjects, checkerboard m-sequence c-VEP
* ``MartinezCagigal2023Pary`` -- 15 subjects, non-binary p-ary m-sequence c-VEP
* ``Thielen2015`` -- 12 subjects, Gold code c-VEP speller
* ``Thielen2021`` -- 30 subjects, pseudo-random c-VEP calculator speller

To run with an alternate dataset:

.. code-block:: bash

   neuralbench eeg cvep --datasets thielen2015

References
~~~~~~~~~~

