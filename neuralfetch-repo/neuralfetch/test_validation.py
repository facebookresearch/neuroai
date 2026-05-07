# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the study validation framework."""

from pathlib import Path

import pytest

from neuralfetch.utils import validation as validation_runner
from neuralfetch.utils.validation import StudyValidation, discover_validations
from neuralfetch.utils.validation.builders import _build_sliding_window
from neuralfetch.utils.validation.config import _clear_cache, _resolve_validation
from neuralfetch.utils.validation.erp import _build_subject_epochs, _evoked_kind_label
from neuralfetch.utils.validation.plots import (
    _plot_drop_grid,
    _plot_group_grand_average_html,
    _short_subject_labels,
)
from neuralset.events import study
from neuralset.events.testing import test2023meg as _test2023meg  # noqa: F401

DISCOVERED = discover_validations()


@pytest.fixture(autouse=True)
def _reset_discovery_cache() -> None:
    """Ensure the discovery cache is fresh for each test module run."""
    _clear_cache()


def test_discover_validations() -> None:
    """discover_validations() should find at least Test2023Meg."""
    found = discover_validations()
    assert len(found) >= 1
    assert "Test2023Meg" in found


@pytest.mark.parametrize("name", list(DISCOVERED))
def test_study_validation_config(name: str) -> None:
    """Check that every discovered validation config has required fields populated."""
    v = DISCOVERED[name]
    assert isinstance(v, StudyValidation)
    assert v.event_type, f"{name}: event_type must not be empty"
    assert v.neuro, f"{name}: neuro config must not be empty"
    assert v.extractor, f"{name}: extractor config must not be empty"
    assert v.model, f"{name}: model config must not be empty"
    assert v.scoring, f"{name}: scoring config must not be empty"
    assert v.mode in ("decod", "encod"), f"{name}: invalid mode"
    assert v.stop > v.start, f"{name}: stop must be > start"


def test_resolve_validation_missing() -> None:
    """_resolve_validation raises ValueError for unknown studies."""
    with pytest.raises((ValueError, ImportError)):
        _resolve_validation("NonExistent9999")


def test_build_sliding_window(tmp_path: Path) -> None:
    """_build_sliding_window constructs a SlidingWindow from Test2023Meg's config."""
    from neuralyze import SlidingWindow

    v = DISCOVERED["Test2023Meg"]
    sw = _build_sliding_window("Test2023Meg", v, tmp_path)
    assert isinstance(sw, SlidingWindow)
    assert sw.mode == v.mode


def _make_synthetic_scores(n_subjects: int = 3, n_times: int = 4):
    """Helper: small xarray of SlidingWindow-shaped synthetic scores."""
    import numpy as np
    import xarray

    subjects = [f"Test2023Meg/{i}" for i in range(n_subjects)]
    data = np.linspace(0.0, 0.5, n_subjects * n_times).reshape(
        n_subjects, 1, 1, 1, 1, 1, n_times
    )
    return xarray.DataArray(
        data=data,
        dims=[
            "subject",
            "split",
            "dim",
            "train_shift",
            "test_shift",
            "test_time",
            "train_time",
        ],
        coords={
            "subject": subjects,
            "split": [0],
            "dim": [0],
            "train_shift": [0],
            "test_shift": [0],
            "test_time": [0],
            "train_time": list(np.linspace(0.0, 0.3, n_times)),
        },
    )


def test_generate_mne_report(tmp_path: Path) -> None:
    """generate_mne_report creates an HTML file with rich metadata."""
    study_cls = study.STUDIES["Test2023Meg"]
    v = DISCOVERED["Test2023Meg"]
    scores = _make_synthetic_scores(n_subjects=1, n_times=3)
    out = validation_runner.generate_mne_report(
        study_cls, v, scores, tmp_path / "report.html"
    )
    assert out.exists()
    assert out.suffix == ".html"
    content = out.read_text("utf-8")
    assert "Test2023Meg" in content
    assert "neuralset" in content
    assert "Peak Score" in content
    assert "Generated at" in content


def test_evoked_kind_label_meg_vs_eeg() -> None:
    """_evoked_kind_label returns ERF for MEG, ERP for EEG/iEEG, Evoked otherwise."""
    import mne

    eeg_info = mne.create_info(
        ch_names=["E1", "E2"], sfreq=100.0, ch_types=["eeg", "eeg"]
    )
    meg_info = mne.create_info(
        ch_names=["M1", "M2"], sfreq=100.0, ch_types=["mag", "grad"]
    )

    assert _evoked_kind_label("Meg", eeg_info) == "ERF"
    assert _evoked_kind_label("Eeg", eeg_info) == "ERP"
    assert _evoked_kind_label("Ieeg", eeg_info) == "ERP"
    assert _evoked_kind_label("", meg_info) == "ERF"
    assert _evoked_kind_label("", eeg_info) == "ERP"
    misc_info = mne.create_info(ch_names=["x"], sfreq=100.0, ch_types=["misc"])
    assert _evoked_kind_label("Something", misc_info) == "Evoked"


def test_build_subject_epochs_returns_epochs(tmp_path: Path) -> None:
    """_build_subject_epochs produces epochs and the ERF label for Test2023Meg."""
    import mne

    study_cls = study.STUDIES["Test2023Meg"]
    study_instance = study_cls(path=tmp_path / "study")
    events = study_instance.run()
    v = DISCOVERED["Test2023Meg"]

    result = _build_subject_epochs(study_instance, events, v, subject_id="Test2023Meg/0")
    assert result is not None
    epochs, label = result
    assert isinstance(epochs, mne.BaseEpochs)
    assert len(epochs) > 0
    assert abs(epochs.tmin - v.start) < 1e-6
    assert label == "ERF"


def test_build_subject_epochs_applies_baseline_when_pretrigger_available(
    tmp_path: Path,
) -> None:
    """Baseline is applied iff the decoding window extends below t=0."""
    study_cls = study.STUDIES["Test2023Meg"]
    study_instance = study_cls(path=tmp_path / "study")
    events = study_instance.run()

    # Default Test2023Meg has start=0.0 -> no baseline should be applied.
    v_no_baseline = DISCOVERED["Test2023Meg"]
    result = _build_subject_epochs(
        study_instance, events, v_no_baseline, subject_id="Test2023Meg/0"
    )
    assert result is not None
    epochs_no_bl, _ = result
    assert epochs_no_bl.baseline is None

    # With a negative start, pre-trigger baseline should be applied.
    v_with_baseline = v_no_baseline.model_copy(update={"start": -0.1})
    result = _build_subject_epochs(
        study_instance, events, v_with_baseline, subject_id="Test2023Meg/0"
    )
    assert result is not None
    epochs_bl, _ = result
    assert epochs_bl.baseline is not None
    assert epochs_bl.baseline[1] == 0.0


def test_plot_drop_grid_marks_excluded_subject() -> None:
    """_plot_drop_grid marks an excluded subject's whole row as bad."""
    import numpy as np

    subject_channels = {
        "sub-A": ["C1", "C2", "C3"],
        "sub-B": ["C1", "C2", "C3"],
    }
    subject_bads = {"sub-A": ["C2"], "sub-B": []}
    fig = _plot_drop_grid(
        subject_channels=subject_channels,
        subject_bads=subject_bads,
        excluded_subjects={"sub-X"},
        all_subjects={"sub-A", "sub-B", "sub-X"},
    )
    axes = fig.axes
    assert axes, "expected at least one axis on the drop grid figure"
    images = axes[0].get_images()
    assert images, "expected an AxesImage on the drop grid"
    matrix = np.asarray(images[0].get_array())
    ylabels = [t.get_text() for t in axes[0].get_yticklabels()]
    xlabels = [t.get_text() for t in axes[0].get_xticklabels()]
    assert ylabels == ["sub-A", "sub-B", "sub-X"]
    assert xlabels == ["C1", "C2", "C3"]
    assert matrix[ylabels.index("sub-X")].tolist() == [1.0, 1.0, 1.0]
    # sub-A bad list is {C2}
    a_row = matrix[ylabels.index("sub-A")].tolist()
    assert a_row == [0.0, 1.0, 0.0]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_group_plot_html_contains_plotly() -> None:
    """_plot_group_grand_average_html embeds a Plotly payload + toggle UI."""
    v = DISCOVERED["Test2023Meg"]
    scores = _make_synthetic_scores(n_subjects=2, n_times=5)
    html = _plot_group_grand_average_html(scores, v)
    assert "Plotly" in html
    assert "Grand Average" in html
    assert "Hide participants" in html
    assert 'id="ga-plotly-' in html
    assert 'id="ga-toggle-' in html


def test_short_subject_labels_strips_study_prefix_and_zero_pads() -> None:
    """Study prefix is stripped; purely-numeric suffixes are zero-padded."""
    ids = [f"Grootswagers2022Human/{i}" for i in range(12)]
    short = _short_subject_labels(ids)
    assert short[0] == "00"
    assert short[9] == "09"
    assert short[10] == "10"
    # Non-numeric suffixes should be left alone (beyond prefix stripping).
    ids_nn = ["SomeStudy/sub-A01", "SomeStudy/sub-B02"]
    assert _short_subject_labels(ids_nn) == ["sub-A01", "sub-B02"]
    # IDs without a slash should pass through unchanged (aside from
    # numeric zero-padding when applicable).
    assert _short_subject_labels(["1", "2", "10"]) == ["01", "02", "10"]


def test_validate_study_report_has_new_sections(tmp_path: Path) -> None:
    """End-to-end: generate_mne_report includes the drop grid, group plot, and summary sections."""
    study_cls = study.STUDIES["Test2023Meg"]
    study_instance = study_cls(path=tmp_path / "study")
    events = study_instance.run()
    # Enable QC drop grid so "Participants x Channels" appears in the report.
    v = DISCOVERED["Test2023Meg"].model_copy(update={"show_qc": True})
    scores = _make_synthetic_scores(n_subjects=2, n_times=4)

    out = validation_runner.generate_mne_report(
        study_cls,
        v,
        scores,
        tmp_path / "report.html",
        study_instance=study_instance,
        events=events,
    )
    assert out.exists()
    content = out.read_text("utf-8")
    assert "Group Grand Average" in content
    assert "Participants x Channels" in content
    assert "Results Summary" in content


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    """`neuralfetch validate --list` should print validatable studies."""
    from neuralfetch.cli import main

    main(["validate", "--list"])
    captured = capsys.readouterr()
    assert "Test2023Meg" in captured.out
