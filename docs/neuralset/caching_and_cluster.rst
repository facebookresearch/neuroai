Caching & Cluster Execution
===========================

NeuralSet pipelines involve expensive computation — loading neural
recordings, running deep models on every stimulus, aligning timelines.
The :term:`infra` parameter adds **disk caching** and optional **cluster
dispatch** to any pipeline component.  Three infra types handle different
granularities:

- **Backend** — caches a whole :term:`Step` result.
- **MapInfra** — caches one result per item in a sequence.
- **TaskInfra** — caches a full computation (training runs, benchmarks).

All three are pydantic models — pass a dict and pydantic instantiates
the right type automatically.
`exca <https://facebookresearch.github.io/exca/>`_ provides the
underlying implementation.

----

Backend
-------

:term:`Study`, :term:`EventsTransform`, and :term:`Chain` are all
:term:`Steps <Step>` — pydantic models with a ``_run()`` method and an
optional ``infra`` field.  When ``infra`` is set, ``run()`` caches the
result of ``_run()`` to disk and can dispatch execution to a cluster.

.. code-block:: python

   import neuralset as ns

   class Double(ns.Step):
       x: int

       def _run(self):
           return self.x * 2

   # No infra — runs every time
   Double(x=5).run()  # 10

   # With caching — second call reads from disk
   Double(x=5, infra={"backend": "Cached", "folder": "/tmp/demo"}).run()

   # On a Slurm cluster
   Double(x=5, infra={"backend": "Slurm", "folder": "/cache",
                       "gpus_per_node": 1}).run()

A :term:`Chain` is itself a :term:`Step` that sequences multiple steps —
each step's output feeds the next.  Each step can carry its own
``infra``; the chain resumes from the **latest cached intermediate**:

.. code-block:: python

   class AddOne(ns.Step):
       def _run(self, value):
           return value + 1

   chain = ns.Chain(steps=[
       Double(x=5, infra={"backend": "Cached", "folder": "/cache"}),
       AddOne(infra={"backend": "Cached", "folder": "/cache"}),
   ])
   chain.run()  # Double is cached; if AddOne changes, only it re-runs

Setting ``infra`` on the chain itself makes the **whole chain run as a
single Slurm job**.

----

MapInfra
--------

:term:`Extractors <Extractor>` have a ``MapInfra`` on their ``infra``
field.  It wraps a method that processes a **sequence of items**,
caching each result independently.  It supports batch dispatch
(processpool, Slurm) so items can be processed in parallel.

.. code-block:: python

   import exca
   import typing as tp

   class Squarer(ns.base._Module):
       coeff: float = 1.0
       infra: exca.MapInfra = exca.MapInfra()

       @infra.apply(item_uid=str)
       def process(self, items: tp.Sequence[int]) -> tp.Iterator[float]:
           for item in items:
               yield item ** 2 * self.coeff

   # Local, no caching
   list(Squarer().process([1, 2, 3]))  # [1.0, 4.0, 9.0]

   # With caching — each item cached individually
   list(Squarer(infra={"folder": "/cache"}).process([1, 2, 3]))

   # Dispatch to Slurm — items batched across jobs
   list(Squarer(infra={"folder": "/cache", "cluster": "slurm",
                        "max_jobs": 64}).process(range(10000)))

----

TaskInfra
---------

NeuralTrain experiments use ``TaskInfra``.  It caches a **complete
computation** — unlike ``MapInfra`` it does not iterate over items.  It
supports Slurm job arrays for parameter sweeps.

.. code-block:: python

   import exca

   class Train(ns.base._Module):
       lr: float = 1e-3
       infra: exca.TaskInfra = exca.TaskInfra()

       @infra.apply
       def run(self):
           return {"loss": 0.42 / self.lr}

   # With caching
   Train(lr=1e-3, infra={"folder": "/results"}).run()

   # On Slurm
   Train(lr=1e-3, infra={"folder": "/results", "cluster": "slurm"}).run()

----

In NeuralSet
------------

Studies — Backend + MapInfra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`Studies <Study>` combine both types.  ``infra`` (``Backend``)
caches the merged events DataFrame, while ``infra_timelines``
(``MapInfra``) caches each timeline independently and defaults to
``cluster="processpool"`` for parallel I/O.

Setting ``infra.folder`` automatically propagates to
``infra_timelines.folder``, so a single ``infra`` is usually enough:

.. code-block:: python

   import neuralset as ns

   study = ns.Study(
       name="Allen2022MassiveSample",
       path=ns.CACHE_FOLDER,
       infra={"backend": "Cached", "folder": "/cache"},
   )
   events = study.run()   # first call: loads and caches
   events = study.run()   # subsequent: reads from cache

During development, disable parallel timeline loading to get clear
tracebacks instead of ``BrokenProcessPool`` errors:

.. code-block:: python

   study = ns.Study(
       name="Allen2022MassiveSample",
       path=ns.CACHE_FOLDER,
       infra_timelines={"cluster": None},
   )

Transforms and Chains — Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`EventsTransforms <EventsTransform>` enrich or filter the events
DataFrame — they are Steps too, and mostly used inside a :term:`Chain`.
Some transforms are GPU-intensive: ``AddSummary`` loads a Llama model to
summarize texts.

Set ``infra`` on individual steps — cache the study locally and dispatch
``AddSummary`` to a GPU node:

.. code-block:: python

   chain = ns.Chain(steps=[
       {"name": "Allen2022MassiveSample", "path": str(ns.CACHE_FOLDER),
        "infra": {"backend": "Cached", "folder": "/cache"}},
       {"name": "AddSummary", "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "infra": {"backend": "Slurm", "folder": "/cache",
                  "gpus_per_node": 1, "partition": "gpu"}},
       {"name": "QueryEvents", "query": "timeline_index < 12"},
   ])
   events = chain.run()

To cache and dispatch a whole chain at once, set ``infra`` on the
chain itself:

.. code-block:: python

   chain = ns.Chain(
       steps=[
           {"name": "Allen2022MassiveSample", "path": str(ns.CACHE_FOLDER),
            "infra": {"backend": "Cached", "folder": "/cache"}},
           {"name": "AddSummary", "model_name": "meta-llama/Llama-3.2-3B-Instruct"},
       ],
       infra={"backend": "Slurm", "folder": "/cache",
              "gpus_per_node": 1, "partition": "gpu"},
   )
   events = chain.run()

Extractors — MapInfra
^^^^^^^^^^^^^^^^^^^^^

:term:`Extractors <Extractor>` use ``MapInfra`` to cache **per event**.
The :term:`prepare` method precomputes results for all events in one
pass — :class:`~neuralset.dataloader.Segmenter` calls it automatically.

Local caching, then scaling to a cluster:

.. code-block:: python

   # Cache locally
   # Requires: `pip install transformers` (or `pip install "neuralset[all]"`)
   extractor = ns.extractors.HuggingFaceImage(
       model_name="facebook/dinov2-small",
       infra={"folder": "/cache"},
   )
   extractor.prepare(events)   # computes & caches per event
   vec = extractor(events, start=0, duration=1.0)  # reads from cache

   # Dispatch to Slurm — events batched across GPU workers
   # Requires: `pip install transformers` (or `pip install "neuralset[all]"`)
   extractor = ns.extractors.HuggingFaceText(
       model_name="gpt2",
       infra={"cluster": "slurm", "folder": "/cache",
              "gpus_per_node": 1, "timeout_min": 120},
   )
   extractor.prepare(events)   # submits jobs, waits for results

----

NeuralTrain experiments
-----------------------

NeuralTrain uses ``TaskInfra`` for full training runs.
:class:`~neuraltrain.utils.BaseExperiment` provides the base pattern:

.. code-block:: python

   from neuraltrain import utils

   class MyExperiment(utils.BaseExperiment):
       lr: float = 1e-3
       # infra: TaskInfra inherited from BaseExperiment

       @infra.apply
       def run(self):
           ...  # training loop

   # Single run with caching
   xp = MyExperiment(lr=1e-3, infra={"folder": "/results", "cluster": "slurm"})
   xp.run()

   # Parameter sweep — each config becomes one Slurm job
   utils.run_grid(MyExperiment, "lr_sweep",
            base_config={"infra": {"folder": "/results", "cluster": "slurm"}},
            grid={"lr": [1e-4, 1e-3, 1e-2]})

``infra.uid_folder()`` returns the cache directory for a specific
configuration — use it for saving checkpoints and artifacts.

----

Configuration reference
-----------------------

These fields are shared across all three infra types.

``folder``
   Directory where cached results are stored.  ``None`` (default)
   disables caching.

``mode``
   Controls cache behavior:

   - ``"cached"`` *(default)* — use cache if available, compute otherwise.
   - ``"force"`` — recompute and overwrite.
   - ``"retry"`` — recompute only if previous run errored.
   - ``"read-only"`` — read from cache; error if missing.

``keep_in_ram``
   Keep results in memory after loading from disk.  Useful for data
   accessed repeatedly; disable for large tensors.

Execution backends
^^^^^^^^^^^^^^^^^^

``Backend`` selects the execution target via the ``backend`` key:

- ``"Cached"`` — local execution with disk caching.
- ``"Slurm"`` — Slurm cluster (supports ``gpus_per_node``,
  ``partition``, ``timeout_min``, etc.).
- ``"SubmititDebug"`` — simulates Slurm locally (same pickling path,
  useful for debugging serialization issues).

``MapInfra`` and ``TaskInfra`` use the ``cluster`` key instead:
``"slurm"``, ``"processpool"``, ``"auto"``, or ``None`` (sequential).
Both accept Slurm parameters: ``slurm_partition``, ``gpus_per_node``,
``timeout_min``, ``cpus_per_task``, ``slurm_constraint``, etc.

Cache management
^^^^^^^^^^^^^^^^

**Backend** (Steps / Chains):

.. code-block:: python

   step.has_cache()      # True if result is cached
   step.clear_cache()    # remove cached result

**TaskInfra**:

.. code-block:: python

   xp.infra.uid_folder()       # Path to this config's cache directory
   xp.infra.iter_cached()      # iterate over cached configs in the folder

**MapInfra**:

.. code-block:: python

   extractor.infra.uid_folder()   # Path to the cache directory
   extractor.infra.cache_dict     # dict-like access to cached items

Set ``mode="force"`` to recompute, or ``mode="read-only"`` in
analysis scripts to prevent accidental recomputation.

**Cache invalidation** is automatic: changing any configuration field
that participates in the UID produces a new cache entry.  Old
entries are not deleted — remove them manually or with ``clear_cache()``.

----

Quick reference
---------------

.. list-table::
   :header-rows: 1

   * - Component
     - ``infra`` type
     - Typical configuration
   * - :term:`Study`
     - ``Backend``
     - ``{"backend": "Cached", "folder": "/cache"}``
   * - Study timelines
     - ``MapInfra``
     - ``{"cluster": "processpool"}`` *(default)*
   * - :term:`Extractor`
     - ``MapInfra``
     - ``{"folder": "/cache", "cluster": "slurm", "gpus_per_node": 1}``
   * - :term:`Chain`
     - ``Backend``
     - ``{"backend": "Cached", "folder": "/cache"}``
   * - :term:`EventsTransform`
     - ``Backend``
     - ``{"backend": "Cached", "folder": "/cache"}``
   * - NeuralTrain experiment
     - ``TaskInfra``
     - ``{"folder": "/results", "cluster": "slurm"}``

.. seealso::

   `exca documentation <https://facebookresearch.github.io/exca/>`_
   for the full API reference (all backend options, Slurm parameters,
   UID computation, custom cache types).

.. raw:: html

   <div class="page-nav">
     <a href="auto_examples/walkthrough/index.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorials</a>
     <a href="encoding_decoding.html" class="page-nav-btn">Encoding &amp; Decoding &rarr;</a>
   </div>
