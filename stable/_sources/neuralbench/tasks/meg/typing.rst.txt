Typing decoding
===============

| **Name**: typing
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Levy2026NoninvasiveMeg`
| **Objective** :bdg-info:`Multiclass classification`
| **Split**: Leave-sentences-out

Usage
~~~~~

.. code-block:: bash

   neuralbench meg typing

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/meg/typing/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The typing decoding task involves decoding the characters that were typed on a computer keyboard while MEG was recorded [Levy2026]_. In this task, we use the public Levy2026NoninvasiveMeg dataset, which contains MEG data recorded while subjects typed back sentences that were shown on a screen.

Dataset Notes
~~~~~~~~~~~~~

* Train/test splits are created by clustering similar sentences together into the same split to avoid data leakage (see `neuralset.splitting.SimilaritySplitter`).

References
~~~~~~~~~~

.. [Levy2026] Lévy, Jarod, et al. "Noninvasive decoding of typed sentences from human brain activity." Nature Neuroscience (2026).
