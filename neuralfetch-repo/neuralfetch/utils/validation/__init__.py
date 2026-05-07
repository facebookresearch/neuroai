# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Study validation runner.

Run a study's validation analysis using :mod:`neuralyze` and generate
an MNE Report (interactive HTML).  The CLI wiring lives in
:mod:`neuralfetch.cli.validate`; this module is library-only.
"""

from .config import (
    StudyValidation,
    TRFConfig,
    discover_validations,
    list_validatable_studies,
)
from .report import generate_mne_report
from .runner import validate_study

__all__ = [
    "StudyValidation",
    "TRFConfig",
    "discover_validations",
    "list_validatable_studies",
    "generate_mne_report",
    "validate_study",
]
