Typing decoding
===============

| **Name**: typing
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Levy2026NoninvasiveEeg`
| **Objective** :bdg-info:`Multiclass classification`
| **Split**: Leave-sentences-out

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg typing

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/typing/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The typing decoding task involves decoding the characters that were typed on a computer keyboard while EEG recordings was recorded [Levy2026eeg]_. In this task, we use the public Levy2026NoninvasiveEeg dataset [Levy2026NoninvasiveEeg]_, which contains EEG data recorded while subjects typed back sentences that were shown on a screen.

Dataset Notes
~~~~~~~~~~~~~

* [TO BE UPDATED] Train/test splits are created by clustering similar sentences together into the same split to avoid data leakage (see `neuralset.splitting.SimilaritySplitter`).

References
~~~~~~~~~~

.. [Levy2026eeg] Lévy, Jarod, et al. "Noninvasive decoding of typed sentences from human brain activity." Nature Neuroscience (2026).
.. [Levy2026NoninvasiveEeg] TODO
