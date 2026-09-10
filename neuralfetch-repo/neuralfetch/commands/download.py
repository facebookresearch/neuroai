# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download a registered study's raw dataset.

Usage::

    neuralfetch download Grootswagers2022Human
    neuralfetch download --list

The download runner lives in :mod:`neuralfetch.utils.runner`; this module is
argparse glue only.
"""

from __future__ import annotations

import argparse

NAME = "download"
HELP = "Download a study dataset."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "study",
        nargs="?",
        default=None,
        help="Study name (e.g. Grootswagers2022Human).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Root folder for study data (default: $NEURALSET_STUDY_FOLDER).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the study has already been downloaded.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete the study's download/ and prepare/ folders first, then "
            "re-download from scratch (implies --overwrite; prompts for "
            "confirmation unless --yes is given)."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="assume_yes",
        help="Skip the confirmation prompt for --clean.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_studies",
        help="List all registered studies and exit.",
    )


def run(args: argparse.Namespace) -> None:
    from neuralfetch.utils.runner import download_study, list_downloadable_studies

    if args.list_studies:
        for name in list_downloadable_studies():
            print(f"  {name}")
        return

    if args.study is None:
        raise SystemExit("error: study name required (or pass --list)")

    download_study(
        name=args.study,
        path=args.path,
        overwrite=args.overwrite,
        clean=args.clean,
        assume_yes=args.assume_yes,
    )
