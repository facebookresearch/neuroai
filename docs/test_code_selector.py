"""Smoke-tests for the JSON-driven extractor configs in the docs.

The interactive quickstart (``docs/_static/code-selector.js``) builds Python
snippets from extractor configs in ``docs/_static/code-selector-data.json``.
These tests instantiate every extractor against the installed ``neuralset``
package so pydantic catches typos / kwarg drift.

Run with::

    pytest docs/test_code_selector.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import neuralset as ns

DATA_PATH = Path(__file__).parent / "_static" / "code-selector-data.json"
DATA = json.loads(DATA_PATH.read_text())


@pytest.fixture
def infra(tmp_path: Path) -> dict[str, Any]:
    return {"folder": str(tmp_path), "cluster": None}


@pytest.mark.parametrize("name", sorted(DATA["task"]))
def test_task_stim_instantiates(name: str, infra: dict[str, Any]) -> None:
    src = DATA["task"][name]["stim"]
    exec(src, {"ns": ns, "infra": infra})


@pytest.mark.parametrize("name", sorted(DATA["device"]))
def test_device_neuro_instantiates(name: str, infra: dict[str, Any]) -> None:
    src = DATA["device"][name]["neuro"]
    exec(src, {"ns": ns, "infra": infra})


@pytest.mark.parametrize("name", sorted(DATA["presets"]))
def test_preset_keys_resolve(name: str) -> None:
    p = DATA["presets"][name]
    assert p["taskKey"] in DATA["task"], (
        f"preset {name}: unknown taskKey {p['taskKey']!r}"
    )
    assert p["deviceKey"] in DATA["device"], (
        f"preset {name}: unknown deviceKey {p['deviceKey']!r}"
    )
