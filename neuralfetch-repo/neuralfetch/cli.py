# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""neuralfetch CLI.

Subcommands:

* ``validate`` - run a study's validation analysis and emit an MNE Report.
* ``download`` - download a registered study's raw dataset.

Usage::

    python -m neuralfetch validate Grootswagers2022Human \\
        --output-dir ~/dataset_validations

    # Run the heavy compute on SLURM (60 min wall-clock budget):
    python -m neuralfetch validate Grootswagers2022Human \\
        --output-dir ~/dataset_validations \\
        --cluster slurm --slurm-partition learnfair \\
        --slurm-time-min 60

    python -m neuralfetch validate --list

    python -m neuralfetch download Grootswagers2022Human
    python -m neuralfetch download --list

    python -m neuralfetch study-info Grootswagers2022Human
    python -m neuralfetch study-info Grootswagers2022Human --update

Analysis and download logic live in :mod:`neuralfetch.utils.validation`,
:mod:`neuralfetch.utils.download`, and :mod:`neuralfetch.utils.base`;
this module is argparse glue only.
"""

from __future__ import annotations

import argparse
import logging
import typing as tp

from neuralfetch.utils.base import compute_study_info, format_study_info, update_source_info
from neuralfetch.utils.download import download_study, list_downloadable_studies
from neuralfetch.utils.validation import list_validatable_studies, validate_study

# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "study",
        nargs="?",
        default=None,
        help="Study name (e.g. Grootswagers2022Human).",
    )
    parser.add_argument(
        "--study-folder",
        default=None,
        help="Root folder for study data (default: auto-detect via NEURALSET_STUDY_FOLDER).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the HTML report (required unless --list).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for intermediate results (default: ~/.cache/neuralset/).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Optional pandas query to pre-filter timelines, e.g. "
            "'subject_index < 1' or 'subject_index in [0, 1, 2]' "
            "(uses neuralset.events.utils.query_with_index semantics)."
        ),
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help=(
            "Execution backend passed to exca (e.g. 'slurm', 'local', or "
            "omit for local in-process). Propagates to SlidingWindow, "
            "neuro extractor, and feature extractor."
        ),
    )
    parser.add_argument(
        "--slurm-partition",
        default=None,
        dest="slurm_partition",
        help="SLURM partition (e.g. learnfair, learnaccel, scavenge). Only used when --cluster slurm.",
    )
    parser.add_argument(
        "--slurm-time-min",
        type=int,
        default=None,
        dest="slurm_time_min",
        help=(
            "SLURM wall-time limit in minutes (maps to exca's timeout_min). "
            "Partition defaults are typically 5 min, so set this for any "
            "non-trivial run. Only used when --cluster slurm."
        ),
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        dest="retry",
        help=(
            "Retry any cached errors instead of re-raising them. "
            "Useful after fixing a bug that caused a previous run to fail."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_studies",
        help="List all studies with a validation config and exit.",
    )


def _run_validate(args: argparse.Namespace) -> None:
    if args.list_studies:
        studies = list_validatable_studies()
        if not studies:
            print("No studies with validation configs found.")
            return
        for name, val in sorted(studies.items()):
            print(f"  {name}: {val.description}")
        return

    if args.study is None:
        raise SystemExit("error: study name required (or pass --list)")
    if args.output_dir is None:
        raise SystemExit("error: --output-dir is required")

    infra: dict[str, tp.Any] | None = None
    if (
        args.cluster is not None
        or args.slurm_partition is not None
        or args.slurm_time_min is not None
        or args.retry
    ):
        infra = {}
        if args.cluster is not None:
            infra["cluster"] = args.cluster
        if args.slurm_partition is not None:
            infra["slurm_partition"] = args.slurm_partition
        if args.slurm_time_min is not None:
            infra["timeout_min"] = args.slurm_time_min
        if args.retry:
            infra["retry"] = True

    validate_study(
        name=args.study,
        study_folder=args.study_folder,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        query=args.query,
        infra=infra,
    )


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _add_download_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--list",
        action="store_true",
        dest="list_studies",
        help="List all registered studies and exit.",
    )


def _run_download(args: argparse.Namespace) -> None:
    if args.list_studies:
        for name in list_downloadable_studies():
            print(f"  {name}")
        return

    if args.study is None:
        raise SystemExit("error: study name required (or pass --list)")

    download_study(name=args.study, path=args.path, overwrite=args.overwrite)


# ---------------------------------------------------------------------------
# study-info
# ---------------------------------------------------------------------------


def _add_study_info_arguments(parser: argparse.ArgumentParser) -> None:
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


def _run_study_info(args: argparse.Namespace) -> None:
    from neuralfetch.utils.base import root_study_folder

    folder = args.path or (root_study_folder() / args.study)
    if args.update:
        actual = update_source_info(name=args.study, folder=folder)
        print(f"Updated _info in source file for {args.study}.")
    else:
        actual = compute_study_info(name=args.study, folder=folder)

    print(f"study.{format_study_info(actual)}")


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="neuralfetch")
    subs = parser.add_subparsers(dest="command", required=True)

    p_validate = subs.add_parser(
        "validate",
        help="Run study validation and generate an MNE Report.",
    )
    _add_validate_arguments(p_validate)
    p_validate.set_defaults(func=_run_validate)

    p_download = subs.add_parser(
        "download",
        help="Download a study dataset.",
    )
    _add_download_arguments(p_download)
    p_download.set_defaults(func=_run_download)

    p_study_info = subs.add_parser(
        "study-info",
        help="Compute StudyInfo for a downloaded study.",
    )
    _add_study_info_arguments(p_study_info)
    p_study_info.set_defaults(func=_run_study_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
