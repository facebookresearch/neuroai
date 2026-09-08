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

.. currentmodule:: neuralbench.data

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_default_dataloaders

.. currentmodule:: neuralbench.pl_module

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BrainModule

Running the benchmark
---------------------

Three entry points onto the same experiments -- the ``neuralbench`` command, the
same selections from Python, and one model you built over a selection.  They are
compared in the :doc:`quickstart <auto_examples/quickstart/index>`.

.. currentmodule:: neuralbench

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_benchmark_cli
   run_benchmark
   check_model
   evaluate_model

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
