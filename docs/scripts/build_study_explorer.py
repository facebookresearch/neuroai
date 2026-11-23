# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from neuralfetch._explorer import renderer


def build_docs_study_explorer(path: Path) -> Path:
    """Regenerate the public NeuralFetch docs fragment from StudyInfo metadata."""
    explorer = renderer.StudyInfoSummaries(
        device_colors={
            "Eeg": "#2563eb",
            "Emg": "#16a34a",
            "Fmri": "#dc2626",
            "Fnirs": "#eab308",
            "Ieeg": "#f97316",
            "Meg": "#c026d3",
            "Multimodal": "#111827",
        },
        install_commands={"neuralfetch": "pip install neuralfetch"},
        packages=["neuralfetch"],
        title="NeuralFetch study explorer",
    )
    fragment = explorer.render_html_fragment()
    path.write_text(
        '<div id="neuralfetch-study-explorer">\n' + fragment + "\n</div>\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    docs_root = Path(__file__).resolve().parents[1]
    report_path = build_docs_study_explorer(
        docs_root / "neuralfetch" / "_explore_studies.html"
    )
    print(f"Saved HTML fragment to {report_path}")
