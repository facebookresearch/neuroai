# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for ``python -m neuralfetch.commands``; delegates to :func:`neuralfetch.commands.main`."""

from neuralfetch.commands import main

if __name__ == "__main__":
    main()
