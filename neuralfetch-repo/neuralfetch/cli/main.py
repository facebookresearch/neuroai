# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""neuralfetch CLI.

Subcommands:

* ``download`` - download a registered study's raw dataset.
* ``study-info`` - compute (or update) StudyInfo for a downloaded study.
* ``export-bids`` - export a study to a BIDS directory tree.

Usage::

    neuralfetch download Grootswagers2022Human
    neuralfetch download --list

    neuralfetch study-info Grootswagers2022Human
    neuralfetch study-info Grootswagers2022Human --update

    neuralfetch export-bids Grootswagers2022Human \\
        --output-dir ~/bids/Grootswagers2022Human \\
        --device Eeg --task thingseeg

    # Parallelise across SLURM (one job per timeline):
    neuralfetch export-bids Grootswagers2022Human \\
        --output-dir ~/bids/Grootswagers2022Human \\
        --device Eeg --task thingseeg \\
        --infra-cluster slurm \\
        --infra-folder /tmp/bids_jobs \\
        --infra-slurm-partition learnfair \\
        --infra-slurm-time-min 60

    # With anonymization:
    neuralfetch export-bids Grootswagers2022Human \\
        --output-dir ~/bids/Grootswagers2022Human \\
        --device Eeg --task thingseeg \\
        --anonymize-daysback 365

Study-info helpers come from :mod:`neuralfetch.utils`; download and export
logic live in :mod:`neuralfetch.cli.download` and
:mod:`neuralfetch.cli.bids_exporter`; this module is argparse glue only.
"""

from __future__ import annotations

import argparse
import logging
import typing as tp

from neuralfetch.cli.bids_exporter import MNE_RAW_TYPES, study_to_bids
from neuralfetch.cli.download import download_study, list_downloadable_studies
from neuralfetch.utils import compute_study_info, format_study_info, update_source_info

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


def _run_download(args: argparse.Namespace) -> None:
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
    from neuralfetch.utils import root_study_folder

    folder = args.path or (root_study_folder() / args.study)
    if args.update:
        actual = update_source_info(name=args.study, folder=folder)
        print(f"Updated _info in source file for {args.study}.")
    else:
        actual = compute_study_info(name=args.study, folder=folder)

    print(f"study.{format_study_info(actual)}")


# ---------------------------------------------------------------------------
# export-bids
# ---------------------------------------------------------------------------


def _add_export_bids_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "study",
        help="Study name (e.g. Grootswagers2022Human).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Root directory for the BIDS output.",
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=sorted(MNE_RAW_TYPES),
        help="Neurophysiology recording type (e.g. Eeg, Meg, Ieeg, Emg, Fnirs).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Root folder for study data (default: $NEURALSET_STUDY_FOLDER/<study>).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "BIDS task label. If omitted, the 'task' column in the events "
            "DataFrame is used."
        ),
    )
    parser.add_argument(
        "--anonymize-daysback",
        type=int,
        default=None,
        dest="anonymize_daysback",
        help=(
            "Number of days to subtract from the measurement date for "
            "anonymization. Required to enable anonymization."
        ),
    )
    parser.add_argument(
        "--anonymize-keep-his",
        action="store_true",
        default=False,
        dest="anonymize_keep_his",
        help="Keep hospital information system (HIS) data when anonymizing.",
    )
    parser.add_argument(
        "--anonymize-keep-source",
        action="store_true",
        default=False,
        dest="anonymize_keep_source",
        help="Keep the source file path when anonymizing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BIDS files.",
    )
    # infra / SLURM options (map to exca.MapInfra passed to BidsExporter)
    parser.add_argument(
        "--infra-cluster",
        default=None,
        dest="infra_cluster",
        help=(
            "Compute backend for per-timeline BIDS writes "
            "(e.g. 'slurm', 'processpool'). Defaults to 'processpool'."
        ),
    )
    parser.add_argument(
        "--infra-folder",
        default=None,
        dest="infra_folder",
        help="Cache/job folder for exca (required when using SLURM).",
    )
    parser.add_argument(
        "--infra-slurm-partition",
        default=None,
        dest="infra_slurm_partition",
        help="SLURM partition for per-timeline jobs (e.g. learnfair). Only used with --infra-cluster slurm.",
    )
    parser.add_argument(
        "--infra-slurm-time-min",
        type=int,
        default=None,
        dest="infra_slurm_time_min",
        help="SLURM wall-time limit in minutes per timeline job. Only used with --infra-cluster slurm.",
    )
    parser.add_argument(
        "--infra-mem-gb",
        type=float,
        default=None,
        dest="infra_mem_gb",
        help="Memory in GB per timeline job. Only used with --infra-cluster slurm.",
    )


def _run_export_bids(args: argparse.Namespace) -> None:
    import exca

    import neuralset as ns
    from neuralfetch.utils import root_study_folder

    folder = args.path or (root_study_folder() / args.study)

    anonymize: dict[str, tp.Any] | None = None
    if args.anonymize_daysback is not None:
        anonymize = {"daysback": args.anonymize_daysback}
        if args.anonymize_keep_his:
            anonymize["keep_his"] = True
        if args.anonymize_keep_source:
            anonymize["keep_source"] = True

    infra_bids: exca.MapInfra | None = None
    if args.infra_cluster is not None:
        infra_kwargs: dict[str, tp.Any] = {"cluster": args.infra_cluster}
        if args.infra_folder is not None:
            infra_kwargs["folder"] = args.infra_folder
        if args.infra_slurm_partition is not None:
            infra_kwargs["slurm_partition"] = args.infra_slurm_partition
        if args.infra_slurm_time_min is not None:
            infra_kwargs["timeout_min"] = args.infra_slurm_time_min
        if args.infra_mem_gb is not None:
            infra_kwargs["mem_gb"] = args.infra_mem_gb
        infra_bids = exca.MapInfra(**infra_kwargs)

    study = ns.Study(name=args.study, path=folder)
    bids_root = study_to_bids(
        study=study,
        path=args.output_dir,
        device=args.device,
        task=args.task,
        anonymize=anonymize,
        overwrite=args.overwrite,
        infra_bids=infra_bids,
    )
    print(f"BIDS export complete: {bids_root}")


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="neuralfetch")
    subs = parser.add_subparsers(dest="command", required=True)

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

    p_export_bids = subs.add_parser(
        "export-bids",
        help="Export a study to a BIDS directory tree.",
    )
    _add_export_bids_arguments(p_export_bids)
    p_export_bids.set_defaults(func=_run_export_bids)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
