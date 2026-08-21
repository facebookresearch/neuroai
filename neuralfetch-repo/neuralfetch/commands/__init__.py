# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""neuralfetch command-line interface.

Each subcommand lives in its own module (``neuralfetch.commands.<name>``) and
exposes ``NAME``, ``HELP``, ``add_arguments(parser)``, and ``run(args)``. This
dispatcher registers each as an argparse subparser, so ``neuralfetch -h`` and
``neuralfetch <command> -h`` are generated automatically. Command modules keep
their heavy imports inside ``run()`` so registering the subparsers stays cheap.
"""

from __future__ import annotations

import argparse
import importlib
import logging

# Registered command modules, in the order they appear in ``neuralfetch -h``.
_COMMANDS = ("download", "study_info", "export_bids")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="neuralfetch")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        mod = importlib.import_module(f"neuralfetch.commands.{name}")
        sub = subs.add_parser(mod.NAME, help=mod.HELP, description=mod.__doc__)
        mod.add_arguments(sub)
        sub.set_defaults(func=mod.run)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
