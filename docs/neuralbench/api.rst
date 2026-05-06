API Reference
=============

Core
----

.. currentmodule:: neuralbench.main

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Data
   Experiment
   BenchmarkAggregator

.. currentmodule:: neuralbench.pl_module

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BrainModule

CLI
---

.. currentmodule:: neuralbench.cli

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_benchmark
   run_benchmark_cli

Events Transforms
-----------------

.. currentmodule:: neuralbench.transforms

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TextPreprocessor
   SklearnSplit
   SimilaritySplit
   PredefinedSplit
   CropSleepRecordings
   CropTimelines
   AddDefaultEvents
   OffsetEvents

Callbacks
---------

.. currentmodule:: neuralbench.callbacks

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TestFullRetrievalMetrics
   RecordingLevelEval
   PlotConfusionMatrix
   PlotRegressionVectors
   plot_confusion_matrix

Utilities
---------

.. currentmodule:: neuralbench.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TrainerConfig
   load_checkpoint
   compute_class_weights_from_dataset
   make_weighted_sampler

Modules
-------

.. currentmodule:: neuralbench.modules

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DownstreamWrapper
   DownstreamWrapperModel

Configuration
-------------

.. currentmodule:: neuralbench.config_manager

.. autosummary::
   :toctree: generated/
   :nosignatures:

   setup_config
   load_config
   get_config
