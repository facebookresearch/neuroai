# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download the pretraining corpus.

Training reads what is on disk: ``Study.run()`` never fetches, only
``Study.download()`` does. Run this once per machine before
``ssl_example.grids.defaults``. The corpus is ~1.1 TB, so check ``DATADIR``
has room first.
"""

import neuralset as ns

from .defaults import DATADIR, STUDIES  # type: ignore


def download() -> None:
    """Fetch every study of ``STUDIES``, skipping what is already on disk."""
    for name in STUDIES:
        print(f"--- {name} -> {DATADIR}")
        ns.Study(name=name, path=DATADIR).download()


if __name__ == "__main__":
    download()
