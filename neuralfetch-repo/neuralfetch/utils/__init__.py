# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for neuralfetch study development.

The public helpers are re-exported here so ``from neuralfetch.utils import ...``
keeps working. The BIDS exporter and download runner live in the
:mod:`neuralfetch.utils.bids` and :mod:`neuralfetch.utils.runner` submodules and
are imported lazily (they pull in heavy dependencies).
"""

from neuralfetch.utils.data import (
    add_sentences,
    download_things_images,
    ensure_imagenet22k,
    scan_files,
)
from neuralfetch.utils.study_info import (
    compute_study_info,
    format_study_info,
    root_study_folder,
    update_source_info,
)

__all__ = [
    "add_sentences",
    "compute_study_info",
    "download_things_images",
    "ensure_imagenet22k",
    "format_study_info",
    "root_study_folder",
    "scan_files",
    "update_source_info",
]
