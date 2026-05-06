Typing decoding
===============

| **Name**: typing
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Levy2025BrainMeg`
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

The typing decoding task involves decoding the characters that were typed on a computer keyboard while MEG was recorded [Levy2025]_. In this task, we use the private Levy2025BrainMeg dataset, which contains MEG data recorded while subjects typed back sentences that were shown on a screen.

Dataset Notes
~~~~~~~~~~~~~

* Train/test splits are created by clustering similar sentences together into the same split to avoid data leakage (see `neuralset.splitting.SimilaritySplitter`).

References
~~~~~~~~~~

.. [Levy2025] Lévy, Jarod, et al. "Brain-to-text decoding: A non-invasive approach via typing." arXiv preprint arXiv:2502.17480 (2025).
