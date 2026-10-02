Word decoding
=============

| **Name**: word
| **Category**: cognitive decoding
| **Dataset**: :py:class:`~neuralset.studies.Nieuwland2018`
| **Objective**: :bdg-dark:`Retrieval`
| **Split**: Predefined

Usage
~~~~~

.. code-block:: bash

   neuralbench eeg word

.. dropdown:: Show ``config.yaml``

   .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/word/config.yaml
      :language: yaml


Description
~~~~~~~~~~~

The word decoding task involves decoding word stimuli from EEG recordings [dAscoli2025]_. In this task, we use the Nieuwland2018 dataset [Nieuwland2018]_, which contains EEG data recorded across 8 UK laboratories while subjects read 80 sentences on a screen in a rapid serial visual presentation paradigm. Word embeddings are extracted using contextualized GPT-2 representations.

We exclude the GLAS (Glasgow, 128-channel BioSemi) and LOND (London, 34-channel) sites because their EEG montages are incompatible with the standard ~64-channel 10-20 systems used by the other 6 sites. Including them inflates the channel dimension to 194 (the union of all unique channel names) with heavy zero-padding, significantly slowing training without improving evaluation quality.

As in [dAscoli2025]_, the retrieval set is built from the 250 most frequent words in the test split.

References
~~~~~~~~~~

.. [dAscoli2025] d'Ascoli, Stéphane, et al. "Towards decoding individual words from non-invasive brain recordings." Nature Communications 16.1 (2025): 10521.
.. [Nieuwland2018] Nieuwland, Mante S., et al. "Large-scale replication study reveals a limit on probabilistic prediction in language comprehension." ELife 7 (2018): e33468.
