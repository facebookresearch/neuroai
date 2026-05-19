"""Generate ``docs/_static/code-builder-data.js`` from the YAML source.

Wired into Sphinx via ``app.connect("builder-inited", ...)`` in ``docs/conf.py``,
and called directly by ``docs/test_code_builder.py``. Pure function, no I/O
side-effects beyond the two passed-in paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HEADER = (
    "/* AUTO-GENERATED from docs/_data/code-builder-data.yaml. DO NOT EDIT. */\n"
    "window.CB_DATA = "
)


def load(yaml_path: Path) -> dict:
    return yaml.safe_load(Path(yaml_path).read_text())


def render(data: dict) -> str:
    return HEADER + json.dumps(data, indent=2) + ";\n"


def build(yaml_path: Path, js_path: Path) -> None:
    Path(js_path).write_text(render(load(yaml_path)))


if __name__ == "__main__":
    here = Path(__file__).parent
    build(
        here / "code-builder-data.yaml",
        here.parent / "_static" / "code-builder-data.js",
    )
    print(f"Wrote {here.parent / '_static' / 'code-builder-data.js'}")
