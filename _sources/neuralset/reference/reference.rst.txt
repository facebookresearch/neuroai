API Reference
=============

Core
----

.. currentmodule:: neuralset.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseModel
   Step
   Chain
   Frequency
   TimedArray

Studies
-------

.. currentmodule:: neuralset.events.study

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Study
   SpecialLoader
   StudyInfo

Events
------

.. currentmodule:: neuralset.events.etypes

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Action
   Artifact
   Audio
   Background
   BaseDataEvent
   BaseSplittableEvent
   BaseText
   CategoricalEvent
   Eeg
   Emg
   Event
   EyeState
   Fmri
   Fnirs
   Ieeg
   Image
   Keystroke
   Meg
   MneRaw
   Phoneme
   Seizure
   Sentence
   SleepStage
   Spikes
   Stimulus
   Text
   Video
   Word

Transforms
----------

.. currentmodule:: neuralset.events.transforms.basic

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AlignEvents
   ConfigureEventLoader
   CreateColumn
   QueryEvents
   RemoveMissing
   SelectIdx

.. currentmodule:: neuralset.events.transforms.text

.. autosummary::
   :toctree: generated/
   :nosignatures:

   EnsureTexts
   AddConcatenationContext
   AddContextToWords
   AddPartOfSpeech
   AddPhonemes
   AddSentenceToWords
   AddSummary

.. currentmodule:: neuralset.events.transforms.splitting

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AssignKSplits
   AssignSentenceSplit
   AssignWordSplitAndContext
   SklearnSplit

.. currentmodule:: neuralset.events.transforms.audio

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ExtractAudioFromVideo

.. currentmodule:: neuralset.events.transforms.chunking

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ChunkEvents

Extractors — Base
-----------------

.. currentmodule:: neuralset.extractors.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseExtractor
   BaseStatic

Extractors — General
--------------------

.. currentmodule:: neuralset.extractors.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   EventDetector
   EventField
   LabelEncoder
   Pulse

Extractors — Neuro
-------------------

.. currentmodule:: neuralset.extractors.neuro

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ChannelPositions
   EegExtractor
   EmgExtractor
   FmriExtractor
   FnirsExtractor
   HrfConvolve
   IeegExtractor
   MegExtractor
   MneRaw
   SpikesExtractor

Extractors — Audio
------------------

.. currentmodule:: neuralset.extractors.audio

.. autosummary::
   :toctree: generated/
   :nosignatures:

   HuggingFaceAudio
   MelSpectrum
   SeamlessM4T
   SonarAudio
   SpeechEnvelope
   Wav2Vec
   Wav2VecBert
   Whisper

Extractors — Text
-----------------

.. currentmodule:: neuralset.extractors.text

.. autosummary::
   :toctree: generated/
   :nosignatures:

   HuggingFaceText
   SentenceTransformer
   SonarEmbedding
   SpacyEmbedding
   TfidfEmbedding
   WordFrequency
   WordLength

Extractors — Image
------------------

.. currentmodule:: neuralset.extractors.image

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ColorHistogram
   HOG
   HuggingFaceImage
   LBP
   RFFT2D

Extractors — Video
------------------

.. currentmodule:: neuralset.extractors.video

.. autosummary::
   :toctree: generated/
   :nosignatures:

   HuggingFaceVideo

Extractors — Meta
-----------------

.. currentmodule:: neuralset.extractors.meta

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AggregatedExtractor
   CroppedExtractor
   ExtractorPCA
   HuggingFacePCA
   TimeAggregatedExtractor
   ToStatic

DataLoader
----------

.. currentmodule:: neuralset.segments

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Segment

.. currentmodule:: neuralset.dataloader

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Segmenter
   SegmentDataset
   Batch

Splitting
---------

.. currentmodule:: neuralset.events.transforms.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DeterministicSplitter
   chunk_events

.. currentmodule:: neuralset.events.transforms.splitting

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SimilaritySplitter
