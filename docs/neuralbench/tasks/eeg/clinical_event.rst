Clinical event detection
========================

| **Name**: clinical event
| **Category**: clinical
| **Dataset**: :py:class:`~neuralset.studies.Harati2015` (TUEV)
| **Objective**: :bdg-primary:`Multilabel classification`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg clinical_event

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/clinical_event/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The goal of the artifact classification task is to predict whether 3-second sliding EEG windows contain one or more of the following event classes [Harati2015]_:

* Artifact ("artf")
* Eye movements ("eyem")
* Generalized periodic epileptiform discharges ("gped")
* Periodic lateralized epileptiform discharges ("pled")
* Spike and slow wave ("spsw")
* Background, i.e. if the event is clearly not any of the other five classes ("bckg")

Of note, we only extract sliding windows that contain at least one of the above event types.

Dataset Notes
~~~~~~~~~~~~~

Previous work (BIOT, LaBraM, CBraMod, etc.) trains and tests on 5-s windows centered around every event contained in the .rec files of the original dataset. Since these events are defined at every second and for every channel that is affected by an event, this means (1) examples can have as much as 4-s of overlap, and (2) each example can be repeated up to C times (where C=21 is the number of EEG channels for which the events are defined). Also, (3), multiple annotations can overlap (e.g. artifact + epileptiform activity at the same time), meaning some windows appear more than once in the training set, but with a different label.

Instead, we suggest a modified framing which addresses issues 2 and 3:
1. Combine consecutive events in .rec such that events now correspond to complete start-end blocks, and overlapping events defined for different channels together
2. Extract sliding windows within the events

Issue 3 is further tackled by framing the problem as a multilabel classification task rather than a multiclass classification task.

References
~~~~~~~~~~

.. [Harati2015] Harati, Amir, et al. "Improved EEG event classification using differential energy." 2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB). IEEE, 2015.
