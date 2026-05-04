# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import modules to ensure configs are registered with DiscriminatedModel
from . import bendr as _bendr  # noqa: F401
from . import common as _common  # noqa: F401
from . import conformer as _conformer  # noqa: F401
from . import conv_transformer as _conv_transformer  # noqa: F401
from . import diffusion_prior as _diffusion_prior  # noqa: F401
from . import dummy_predictor as _dummy_predictor  # noqa: F401
from . import fmri_mlp as _fmri_mlp  # noqa: F401
from . import freqbandnet as _freqbandnet  # noqa: F401
from . import green as _green  # noqa: F401
from . import labram as _labram  # noqa: F401
from . import linear as _linear  # noqa: F401
from . import luna as _luna  # noqa: F401
from . import preprocessor as _preprocessor  # noqa: F401
from . import reve as _reve  # noqa: F401
from . import simpleconv as _simpleconv  # noqa: F401
from . import simplerconv as _simplerconv  # noqa: F401
from . import sit as _sit  # noqa: F401
from . import transformer as _transformer  # noqa: F401
from .base import BaseModelConfig as BaseModelConfig  # used for configs
from .simpleconv import SimpleConvTimeAgg as SimpleConvTimeAgg
