# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Channel positions of EEG recordings from standard montages."""

import mne


def set_bipolar_montage(raw: mne.io.BaseRaw, montage: str = "standard_1005") -> None:
    """Set the positions of ``raw``'s channels from a standard MNE ``montage``.

    A channel named after an electrode of ``montage`` (e.g. ``Fp1``) gets its
    position; a bipolar channel ``A-B`` (e.g. ``Fp1-F7``) gets the position of its
    first electrode ``A``, as ``neuralset.extractors.ChannelPositions`` names it.
    Other channels get no position.
    """
    standard = mne.channels.make_standard_montage(montage).get_positions()
    electrodes = standard["ch_pos"]
    positions = {}
    for name in raw.ch_names:
        first = name.split("-")[0]
        if name in electrodes:
            positions[name] = electrodes[name]
        elif "-" in name and first in electrodes:
            positions[name] = electrodes[first]
    dig = mne.channels.make_dig_montage(
        ch_pos=positions,
        nasion=standard["nasion"],
        lpa=standard["lpa"],
        rpa=standard["rpa"],
        coord_frame=standard["coord_frame"],
    )
    raw.set_montage(dig, on_missing="ignore")
