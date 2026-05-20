# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Tests for the interactive code-builder page (docs/neuralset/code_builder.rst).

Four layers, all driven by fixtures rendered on-demand by the JS
generator (``docs/_data/render_fixtures.mjs``) into a session-scoped
tmp dir (see the ``fixtures_dir`` fixture in ``docs/conftest.py``):

  L0  YAML schema + cls resolution + JS-data-in-sync.  Always runs.
  L1  ``ast.parse`` each rendered ``script.py``.  Always runs.
  L2  Exec only the extractor instantiation lines under real Pydantic
      validation.  Always runs.
  L3  Subprocess-exec each ``script.py`` end-to-end.
      ``@pytest.mark.slow`` — opt-in via ``--slow``.

The JS renderer (``docs/_static/code-builder.js``) is the single source
of truth; the Python tests only consume what it emits, so drift is
impossible by construction. Node must be on PATH for L1/L2/L3.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).parent
DATA_YAML = DOCS / "_data" / "code-builder-data.yaml"
DATA_JS = DOCS / "_static" / "code-builder-data.js"

sys.path.insert(0, str(DOCS / "_data"))
import build_data  # type: ignore  # noqa: E402

DATA = build_data.load(DATA_YAML)
AXES = DATA["axes"]
STUDIES = DATA.get("studies", {}) or {}
DEFAULT_STUDY = DATA["default_study"]
AXIS_ORDER = ["neuro", "stim", "task", "model", "compute", "style"]


# ── Shared helpers ──────────────────────────────────────────────────────────


def _axis_pinned() -> list[dict]:
    """One combo per axis option, with the other axes at their defaults.

    Mirrors the JS-side `axisPinned` in `docs/_data/render_fixtures.mjs`,
    so every fixture rendered there corresponds to exactly one returned
    combo here.
    """
    base = {a: AXES[a]["default"] for a in AXIS_ORDER}
    seen: set[tuple] = set()
    out: list[dict] = []
    for axis in AXIS_ORDER:
        for key in AXES[axis]["options"]:
            sel = {**base, axis: key}
            sig = tuple(sel[a] for a in AXIS_ORDER)
            if sig not in seen:
                seen.add(sig)
                out.append(sel)
    return out


def _id(sel: dict) -> str:
    return "-".join(sel[a] for a in AXIS_ORDER)


def _fixture_script(fixtures_dir: Path, sel: dict) -> str:
    return (fixtures_dir / _id(sel) / "script.py").read_text()


def _extract_extractor_calls(script: str) -> str:
    """Pull the `neuro = ns.extractors.<Cls>(...)` and `stim = ...` assigns
    out of a fixture so L2 can exec just the extractor instantiation
    under real Pydantic validation, without dragging in Study/Segmenter."""
    tree = ast.parse(script)
    chunks: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("neuro", "stim")
            and isinstance(node.value, ast.Call)
        ):
            chunks.append(ast.unparse(node))
    return "\n".join(chunks)


# ── L0: YAML schema + cls resolution + sync checks ──────────────────────────


def test_l0_yaml_axes_match_expected_set() -> None:
    assert set(AXES) == set(AXIS_ORDER), AXES.keys()
    for axis in AXIS_ORDER:
        meta = AXES[axis]
        assert "label" in meta
        assert "options" in meta
        assert "default" in meta
        assert meta["default"] in meta["options"]


def test_l0_neuro_options_have_required_fields() -> None:
    """Every neuro option needs `cls`, `kwargs`, `window`, `event_type`,
    and exactly one of `t` (with optional `t_comment`) or `x_expr`."""
    for key, opt in AXES["neuro"]["options"].items():
        for required in ("cls", "kwargs", "window", "event_type"):
            assert required in opt, f"neuro.{key}: missing {required!r}"
        has_t = "t" in opt
        has_x = "x_expr" in opt
        assert has_t ^ has_x, f"neuro.{key}: must define exactly one of t / x_expr"


def test_l0_stim_options_have_required_fields() -> None:
    for key, opt in AXES["stim"]["options"].items():
        for required in ("cls", "kwargs", "event_type", "is_classification"):
            assert required in opt, f"stim.{key}: missing {required!r}"


def test_l0_extractor_classes_resolve() -> None:
    """Every `cls` string must resolve on `neuralset.extractors`."""
    import neuralset.extractors as nex

    for axis in ("neuro", "stim"):
        for key, opt in AXES[axis]["options"].items():
            cls = getattr(nex, opt["cls"], None)
            assert cls is not None, (
                f"{axis}.{key}.cls={opt['cls']!r} not on neuralset.extractors"
            )


def test_l0_generated_js_is_in_sync() -> None:
    """The committed _static/code-builder-data.js must match what
    build_data would produce now."""
    expected = build_data.render(build_data.load(DATA_YAML))
    actual = DATA_JS.read_text()
    assert expected == actual, (
        "docs/_static/code-builder-data.js is stale. "
        "Regenerate with: python docs/_data/build_data.py"
    )


# ── L1: AST sweep over per-axis-pinned combos ──────────────────────────────


@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l1_rendered_python_parses(sel: dict, fixtures_dir: Path) -> None:
    """Every rendered fixture must be syntactically valid Python."""
    script = _fixture_script(fixtures_dir, sel)
    try:
        ast.parse(script)
    except SyntaxError as e:
        pytest.fail(f"L1 parse failed for {_id(sel)}:\n{e}\n--- fixture ---\n{script}")


# ── L2: per-axis pydantic instantiation ─────────────────────────────────────


@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l2_extractors_instantiate(sel: dict, fixtures_dir: Path, tmp_path: Path) -> None:
    """Exec only the two `neuro = ns.extractors.<Cls>(...)` and
    `stim = ns.extractors.<Cls>(...)` assignments under real Pydantic
    validation. Ensures kwargs in the YAML are accepted by the live
    extractor classes."""
    if sel["compute"] == "slurm":
        pytest.skip("slurm path is generated-only; not exec'd here")
    if sel["style"] == "yaml":
        # YAML-mode buries extractors inside the embedded config string;
        # validation happens at Experiment(**config) time, which the
        # kwargs-mode counterpart already covers via direct instantiation.
        pytest.skip("yaml-mode extractor validation is exercised by L3")

    import neuralset as ns
    import neuralset.extractors  # force submodule load so `ns.extractors` resolves  # noqa: F401

    src = _extract_extractor_calls(_fixture_script(fixtures_dir, sel))
    assert src, f"no extractor assignments found in fixture for {_id(sel)}"

    g: dict = {"ns": ns, "infra": {"folder": str(tmp_path), "cluster": None}}
    try:
        exec(src, g)
    except Exception as e:
        pytest.fail(
            f"L2 extractor instantiation failed for {_id(sel)}:\n{e}\n--- src ---\n{src}"
        )


# ── L3: End-to-end (slow, opt-in) ───────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l3_pipeline_runs_end_to_end(
    sel: dict, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Subprocess-exec each fixture, redirecting CACHE/STUDIES into tmp_path."""
    if sel["compute"] == "slurm":
        pytest.skip("slurm path is generated-only; not exec'd here")

    for axis in ("neuro", "stim"):
        opt = AXES[axis]["options"][sel[axis]]
        if opt.get("l3_skip"):
            pytest.skip(f"{axis}.{sel[axis]}: {opt['l3_skip']}")

    # Real public datasets (Allen2022Massive, Bel2026Petit, etc.) require
    # large network downloads, are subject to upstream availability, and
    # often need credentials — keep L3 to ExampleMultiModal combos so it
    # stays fast and self-contained.
    stu = STUDIES.get(f"{sel['neuro']}-{sel['stim']}", DEFAULT_STUDY)
    if stu["name"] != DEFAULT_STUDY["name"]:
        pytest.skip(f"{_id(sel)}: uses real public dataset {stu['name']!r}")

    script = _fixture_script(fixtures_dir, sel)
    pre = f"from pathlib import Path as _P\n_TMP = _P({str(tmp_path)!r})\n"
    script = script.replace(
        'CACHE = Path.home() / "neuroai_data" / ".cache"',
        'CACHE = _TMP / ".cache"',
    ).replace(
        'STUDIES = Path.home() / "neuroai_data" / "studies"',
        'STUDIES = _TMP / "studies"',
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(pre + script)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"L3 exec failed for {_id(sel)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}\n"
            f"--- script ({script_path}) ---\n{script}"
        )
