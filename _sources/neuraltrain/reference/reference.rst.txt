API Reference
=============

Some parts of ``neuraltrain`` depend on optional packages such as
``braindecode``, ``x_transformers``, ``green``, or ``dalle2_pytorch``.
Install ``neuraltrain-repo/.[models]`` when you want the full set of model
and augmentation modules available.

Core Interfaces
---------------

.. currentmodule:: neuraltrain.models.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseModelConfig

.. currentmodule:: neuraltrain.losses.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseLoss
   BaseTorchLoss

.. currentmodule:: neuraltrain.metrics.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseMetric
   BaseTorchMetric

.. currentmodule:: neuraltrain.optimizers.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseOptimizer
   LightningOptimizer

Optimizers & Schedulers
-----------------------

.. currentmodule:: neuraltrain.optimizers.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseTorchOptimizer
   BaseLRScheduler
   BaseTorchLRScheduler

Models
------

When ``braindecode`` is installed, every model it exports (``EEGNet``,
``ShallowFBCSPNet``, ``Deep4Net``, etc.) is auto-registered as a
``BaseBrainDecodeModel`` subclass on ``neuraltrain.models.base``.
Use ``models.EEGNet(kwargs={...})`` — see :doc:`../install` for an example.

.. currentmodule:: neuraltrain.models.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseBrainDecodeModel

.. currentmodule:: neuraltrain.models.simpleconv

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SimpleConv
   SimpleConvModel
   SimpleConvTimeAgg
   SimpleConvTimeAggModel

.. currentmodule:: neuraltrain.models.simplerconv

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SimplerConv
   SimplerConvModel

.. currentmodule:: neuraltrain.models.transformer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TransformerEncoder

.. currentmodule:: neuraltrain.models.conv_transformer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConvTransformer
   ConvTransformerModel

.. currentmodule:: neuraltrain.models.conformer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Conformer

.. currentmodule:: neuraltrain.models.linear

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Linear
   LinearModel

.. currentmodule:: neuraltrain.models.constant_predictor

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConstantPredictor
   ConstantPredictorModel

.. currentmodule:: neuraltrain.models.fmri_mlp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FmriMlp
   FmriMlpModel
   FmriLinear
   FmriLinearModel

.. currentmodule:: neuraltrain.models.reve

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NtReve

.. currentmodule:: neuraltrain.models.luna

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NtLuna

.. currentmodule:: neuraltrain.models.labram

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NtLabram

.. currentmodule:: neuraltrain.models.freqbandnet

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FreqBandNet
   FreqBandNetModel

.. currentmodule:: neuraltrain.models.green

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Green

.. currentmodule:: neuraltrain.models.diffusion_prior

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DiffusionPrior

.. currentmodule:: neuraltrain.models.preprocessor

.. autosummary::
   :toctree: generated/
   :nosignatures:

   OnTheFlyPreprocessor
   OnTheFlyPreprocessorModel

Shared Building Blocks
----------------------

.. currentmodule:: neuraltrain.models.common

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SubjectLayers
   FourierEmb
   ChannelMerger
   Mlp
   TemporalDownsampling
   LayerScale
   NormDenormScaler
   ChannelDropout
   BahdanauAttention

Losses
------

.. currentmodule:: neuraltrain.losses.losses

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MultiLoss
   ClipLoss
   SigLipLoss

Metrics
-------

.. currentmodule:: neuraltrain.metrics.metrics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   OnlinePearsonCorr
   Rank
   TopkAcc
   TopkAccFromScores
   ImageSimilarity
   GroupedMetric

Augmentations
-------------

.. currentmodule:: neuraltrain.augmentations.augmentations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TrivialBrainAugment
   TrivialBrainAugmentConfig
   BandstopFilterFFT
   BandstopFilterFFTConfig
   ChannelsDropoutConfig
   FrequencyShiftConfig
   GaussianNoiseConfig
   SmoothTimeMaskConfig

Experiment Utilities
--------------------

.. currentmodule:: neuraltrain.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_grid
   CsvLoggerConfig
   WandbLoggerConfig
   WandbInfra
   BaseExperiment
   StandardScaler
   TimedIterator
   convert_to_pydantic
