# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401
# ruff: noqa: F401
# isort: skip_file
# pylint: disable=wrong-import-position

import exca as _exca
from packaging.version import Version as _Version

_XK_MIN = "0.5.20"
if _Version(_exca.__version__) < _Version(_XK_MIN):
    raise RuntimeError(f"neuralset requires exca>={_XK_MIN} (pip install -U exca)")

# main subpackages
from . import events as events
from . import extractors as extractors

# loaded after extractors to avoid circular import / populates ns.events
from .events import transforms as _transf

# step/chain API
from .base import Chain as Chain
from .base import Step as Step

# convenience
from .base import CACHE_FOLDER as CACHE_FOLDER
from .base import BaseModel as BaseModel  # improved pydantic BaseModel

# useful base classes
from .events import Event as Event
from .events import Study as Study
from .events import EventsTransform as EventsTransform
from .dataloader import Segmenter as Segmenter
from .dataloader import SegmentDataset as SegmentDataset
from .segments import Segment as Segment
from .extractors import BaseExtractor as BaseExtractor
