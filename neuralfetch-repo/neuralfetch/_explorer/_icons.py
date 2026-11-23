# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import cache
from importlib.resources import files


@cache
def svg(name: str) -> str:
    return (
        files("neuralfetch._explorer")
        .joinpath("static", "icons", f"{name}.svg")
        .read_text(encoding="utf-8")
        .strip()
    )
