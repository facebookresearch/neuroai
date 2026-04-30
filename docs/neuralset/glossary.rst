Glossary
========

Key concepts in the NeuralSet pipeline, from data source to batched tensors.
See the :doc:`tutorials <auto_examples/walkthrough/index>` for how they fit together.

----

.. glossary::
   :sorted:

   Study
      An **interface to an external dataset** that knows how to iterate
      :term:`timelines <Timeline>` and load :term:`events <Event>` from
      raw data files.  Subclasses implement ``iter_timelines()`` and
      ``_load_timeline_events()``.  No data is bundled — each study
      points to an externally hosted repository.
      All studies can share a single ``path`` — each resolves its own
      subfolder automatically.
      → :class:`~neuralset.Study`
      · :doc:`Tutorial <auto_examples/walkthrough/02_studies>`

   Timeline
      A **recording session or run** within a :term:`Study`, typically
      identified by BIDS entities (``subject``, ``session``, ``task``,
      ``run``).  ``Study.iter_timelines()`` yields one dict per
      timeline; events loaded from it share that timeline identifier.

   Event
      **Something that happens in time**: a stimulus, a neural recording, a
      word.  Core fields: ``start``, ``duration``, ``timeline``.
      Concrete subclasses (``Meg``, ``Image``, ``Word``, ``Fmri``, …)
      add modality-specific fields.  Events are collected into a pandas
      DataFrame for bulk processing.
      → :class:`~neuralset.events.Event`
      · :doc:`Tutorial <auto_examples/walkthrough/01_events>`

   EventsTransform
      A **processing step** that takes an events DataFrame and returns a
      modified one (filtering, chunking, linking word context, …).
      Subclasses override ``_run(events) -> events``.
      → :class:`~neuralset.events.transforms.EventsTransform`
      · :doc:`Tutorial <auto_examples/walkthrough/03_transforms>`

   Extractor
      **Converts events into numeric arrays** for a given time window —
      e.g. ``MegExtractor`` returns MEG time-series at a given frequency.
      → :class:`~neuralset.extractors.BaseExtractor`
      · :doc:`Tutorial <auto_examples/walkthrough/04_extractors>`

   Trigger
      The :term:`Event` that **anchors** a :term:`Segment` — for example a
      word onset or an image presentation.  Selected by the
      ``trigger_query`` parameter of the :term:`Segmenter`.

   Segment
      A **time window** (``start``, ``duration``) within a :term:`Timeline`,
      anchored to a :term:`Trigger`.  Defines which slice of data each
      :term:`Extractor` reads.
      → :class:`~neuralset.segments.Segment`
      · :doc:`Tutorial <auto_examples/walkthrough/05_segmenter>`

   Segmenter
      **Creates segments** from an events DataFrame and wraps them into a
      :term:`SegmentDataset`.  Configured with a trigger query, a time
      window, and a dict of extractors.
      → :class:`~neuralset.dataloader.Segmenter`
      · :doc:`Tutorial <auto_examples/walkthrough/05_segmenter>`

   SegmentDataset
      A **torch Dataset** that pairs :term:`segments <Segment>` with
      :term:`extractors <Extractor>`.  Each ``__getitem__`` reads data
      for one segment across all extractors.
      → :class:`~neuralset.dataloader.SegmentDataset`
      · :doc:`Tutorial <auto_examples/walkthrough/05_segmenter>`

   TimedArray
      A **numpy array with time metadata** (``frequency``, ``start``,
      ``duration``).  Used by dynamic extractors for time-aligned slicing.
      → :class:`~neuralset.base.TimedArray`

   Frequency
      A **float subclass** with helpers for converting between seconds
      and sample indices (``to_ind``, ``to_sec``).
      → :class:`~neuralset.base.Frequency`

   Step
      A **composable pipeline unit** — a pydantic model with a ``_run()``
      method and an optional ``infra`` parameter for caching and cluster
      execution.  Base class for :term:`Study`, :term:`EventsTransform`,
      and :term:`Chain`.
      → :class:`~neuralset.base.Step`
      · :doc:`Caching & Cluster Execution <caching_and_cluster>`

   Chain
      **Sequences** :term:`Steps <Step>` — each step's output feeds the
      next.  Auto-created when a list is assigned to a ``Step``-typed
      field.  Resumes from the latest cached intermediate result.
      → :class:`~neuralset.base.Chain`
      · :doc:`Tutorial <auto_examples/walkthrough/06_chains>`

   prepare
      Method on :term:`extractors <Extractor>` that **precomputes results
      for all events** in one pass, triggering disk caching and optional
      cluster dispatch.  Called automatically by
      :meth:`Segmenter.apply() <neuralset.dataloader.Segmenter.apply>`.
      · :doc:`Caching & Cluster Execution <caching_and_cluster>`

   infra
      Parameter controlling **disk caching and cluster execution**.  Its
      type varies by component: ``Backend`` on :term:`Steps <Step>` and
      :term:`Chains <Chain>` (whole-step caching), ``MapInfra`` on
      :term:`extractors <Extractor>` and ``Study.infra_timelines``
      (per-item caching and batch dispatch), ``TaskInfra`` on
      NeuralTrain experiments (full computation and job arrays).
      · :doc:`Caching & Cluster Execution <caching_and_cluster>`

.. raw:: html

   <div class="page-nav">
     <a href="encoding_decoding.html" class="page-nav-btn page-nav-btn--outline">&larr; Encoding &amp; Decoding</a>
     <a href="extending/contributing.html" class="page-nav-btn">Contributing &rarr;</a>
   </div>
