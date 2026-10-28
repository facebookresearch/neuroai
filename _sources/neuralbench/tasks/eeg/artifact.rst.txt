Artifact classification
=======================

| **Name**: artifact
| **Category**: Others
| **Dataset**: :py:class:`~neuralset.studies.Hamid2020` (TUAR)
| **Objective**: :bdg-primary:`Multilabel classification`
| **Split**: Leave-subjects-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg artifact

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/artifact/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The goal of the artifact classification task is to predict whether 3-second EEG sliding windows contain one or mode of the following five artifact types [Hamid2020]_:

* eye movement ("eyem")
* chewing ("chew")
* shivers ("shiv")
* muscle artifact ("musc")
* electrode-related artifacts, i.e. electrode pop, electrostatic, or lead artifact ("elec")

Of note, the seizure annotations available in this dataset are not used in this task.

Dataset Notes
~~~~~~~~~~~~~

* Following the formulation of the *clinical event* classification task, we adopt a multilabel classification approach to better model overlapping events. See :doc:`clinical event classification <./clinical_event>`.
* For this, we break down the combined artifact events ("eyem_musc", "musc_elec", "eyem_elec", etc.) into two separate events with the same start and duration.

References
~~~~~~~~~~

.. [Hamid2020] Hamid, Ahmed, et al. "The Temple University Artifact Corpus: An annotated corpus of EEG artifacts." 2020 IEEE Signal Processing in Medicine and Biology Symposium (SPMB). IEEE, 2020.
