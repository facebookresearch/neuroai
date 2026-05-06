# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Happy-path test for the non-EEG plot script.

Mocks :meth:`BenchmarkAggregator._collect_results` so no cached results
are required on disk, and mocks the final
:func:`plot_non_eeg_bar_chart` call so the synthetic result dicts don't
need to satisfy the full plotting pipeline.  The test's job is to
confirm that :func:`plot_non_eeg.main` (1) collects results for every
requested device, (2) concatenates them, and (3) forwards them to the
plot function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuralbench.plots import plot_non_eeg as pne


def _patch_aggregator(mocker, collect_side_effect):
    """Replace ``BenchmarkAggregator`` with a fake whose ``_collect_results``
    returns canned values, so the test doesn't depend on which studies
    happen to be installed (e.g. private datasets absent from the public
    ``neuralfetch``)."""
    fake_instances = [
        mocker.MagicMock(
            _collect_results=mocker.MagicMock(return_value=ret),
            loss_to_metric_mapping={"CrossEntropyLoss": "test/bal_acc"},
            output_dir="/tmp",
        )
        for ret in collect_side_effect
    ]
    fake_cls = mocker.MagicMock(side_effect=fake_instances)
    mocker.patch.object(pne, "BenchmarkAggregator", fake_cls)
    mocker.patch.object(pne, "_task_configs_valid", return_value=True)
    return fake_cls, fake_instances


def test_main_concatenates_per_device_results(mocker, tmp_path: Path) -> None:
    meg_result = {"_marker": "meg"}
    fmri_result = {"_marker": "fmri"}

    _, fake_instances = _patch_aggregator(mocker, [[meg_result], [fmri_result]])

    fake_df = pd.DataFrame({"train_fraction": [1.0], "dataset_name": ["x"]})
    mock_build = mocker.patch.object(pne, "build_results_df", return_value=fake_df)
    mocker.patch.object(pne, "filter_core_dataset", side_effect=lambda df: df)
    expected_png = tmp_path / "core_non_eeg_bar_chart.png"
    mock_plot = mocker.patch.object(
        pne, "plot_non_eeg_bar_chart", return_value=expected_png
    )

    written = pne.main(devices=("meg", "fmri"), output_dir=tmp_path)

    assert sum(inst._collect_results.call_count for inst in fake_instances) == 2
    mock_build.assert_called_once()
    results_arg, _ = mock_build.call_args.args
    assert meg_result in results_arg and fmri_result in results_arg

    mock_plot.assert_called_once()
    args, kwargs = mock_plot.call_args
    pd.testing.assert_frame_equal(args[0], fake_df)
    assert args[1] == tmp_path
    assert kwargs["devices"] == ("meg", "fmri")
    assert written == expected_png


def test_main_raises_when_no_results(mocker, tmp_path: Path) -> None:
    _patch_aggregator(mocker, [[], []])

    mock_plot = mocker.patch.object(pne, "plot_non_eeg_bar_chart")
    with pytest.raises(SystemExit, match="No cached results"):
        pne.main(devices=("meg", "fmri"), output_dir=tmp_path)
    mock_plot.assert_not_called()
