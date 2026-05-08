# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC task package.

Task-side Python is just :mod:`charset` — the keystroke-vocabulary
constants referenced by ``!!python/name:`` from the YAML configs.

Everything else lives upstream:

* Study source — :mod:`neuralfetch.studies.emg2qwerty`
* CTC target extractor — :class:`neuralset.extractors.text.KeystrokeSequence`
* CER metric — :class:`neuraltrain.metrics.metrics.CharacterErrorRates`
* SpecAugment callback — :class:`neuralbench.callbacks.SpecAugmentCallback`
  (Lightning hook on the model's spectrogram submodule)
* Band-rotation augmentation —
  :class:`neuraltrain.augmentations.BandRotation` (paired Module + Config)
"""
