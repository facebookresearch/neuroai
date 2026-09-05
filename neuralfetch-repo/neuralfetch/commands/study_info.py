# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute (or update) StudyInfo for a downloaded study.

Usage::

    neuralfetch study-info Grootswagers2022Human
    neuralfetch study-info Grootswagers2022Human --update

The StudyInfo helpers live in :mod:`neuralfetch.utils`; this module is argparse
glue only.
"""

from __future__ import annotations

import argparse

NAME = "study-info"
HELP = "Compute StudyInfo for a downloaded study."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "study",
        help="Study name (e.g. Duan2026OmniIeeg).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Root folder for study data (default: $NEURALSET_STUDY_FOLDER).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the _info ClassVar in the study's source file with the computed values.",
    )


def run(args: argparse.Namespace) -> None:
    from neuralfetch.utils import (
        compute_study_info,
        format_study_info,
        root_study_folder,
        update_source_info,
    )

    folder = args.path or (root_study_folder() / args.study)
    if args.update:
        actual = update_source_info(name=args.study, folder=folder)
        print(f"Updated _info in source file for {args.study}.")
    else:
        actual = compute_study_info(name=args.study, folder=folder)

    print(f"study.{format_study_info(actual)}")
