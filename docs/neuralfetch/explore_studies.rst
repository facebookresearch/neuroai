Explore Studies
===============

Browse NeuralFetch studies from their declared ``StudyInfo`` metadata. The
interactive explorer estimates recording volume, summarizes catalog coverage,
and shows which event types each study exposes without downloading datasets.

.. raw:: html
   :file: _explore_studies.html

Download a Study
----------------

Pass any study name to ``ns.Study`` to download and load it:

.. code-block:: python

   import neuralset as ns

   study = ns.Study(name="Grootswagers2022Human", path="./data")
   study.download()
   events = study.run()

See :doc:`samples` for lightweight datasets with ready-to-paste snippets,
or :doc:`reference/reference` for the full API reference.

.. raw:: html

   <div class="page-nav">
     <a href="auto_examples/index.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorials</a>
     <a href="samples.html" class="page-nav-btn">Sample Datasets &rarr;</a>
   </div>
