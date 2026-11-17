# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mne
import numpy as np
import pytest

from neuralfetch.utils.montage import set_bipolar_montage


@pytest.mark.parametrize("montage", ["standard_1020", "standard_1005"])
def test_bipolar_channels_sit_at_their_first_electrode(montage: str) -> None:
    names = ["Fp1-F7", "Fp1-F3", "Cz", "ECG", "XX-F7"]
    types = ["eeg", "eeg", "eeg", "ecg", "eeg"]
    raw = mne.io.RawArray(
        np.zeros((len(names), 10)), mne.create_info(names, 256.0, types), verbose=False
    )
    set_bipolar_montage(raw, montage)
    # the electrodes in the same (head) frame
    ref = mne.io.RawArray(
        np.zeros((2, 10)), mne.create_info(["Fp1", "Cz"], 256.0, "eeg"), verbose=False
    )
    ref.set_montage(montage)
    locs = {ch["ch_name"]: ch["loc"][:3] for ch in raw.info["chs"]}
    expected = {ch["ch_name"]: ch["loc"][:3] for ch in ref.info["chs"]}
    np.testing.assert_allclose(locs["Fp1-F7"], expected["Fp1"])
    np.testing.assert_allclose(locs["Fp1-F3"], expected["Fp1"])
    np.testing.assert_allclose(locs["Cz"], expected["Cz"])
    # no known first electrode: no position
    assert np.isnan(locs["XX-F7"]).all()
