# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export a study to a BIDS directory tree.

Usage::

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

The export logic lives in :mod:`neuralfetch.utils.bids`; this module is argparse
glue only.
"""

from __future__ import annotations

import argparse
import typing as tp

NAME = "export-bids"
HELP = "Export a study to a BIDS directory tree."

# Devices supported by the BIDS exporter (mirrors
# ``neuralfetch.utils.bids.MNE_RAW_TYPES``). Defined here so registering the
# subparser does not import mne/exca; runtime validation happens in
# ``study_to_bids``.
_DEVICES = ("Eeg", "Emg", "Fnirs", "Ieeg", "Meg")


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
        choices=_DEVICES,
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


def run(args: argparse.Namespace) -> None:
    import exca

    import neuralset as ns
    from neuralfetch.utils import root_study_folder
    from neuralfetch.utils.bids import study_to_bids

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
