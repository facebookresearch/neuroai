"""Validate the Python code strings emitted by code-selector.js.

This is a standalone validator (not a pytest test — requires `node` on PATH,
which isn't guaranteed in CI). It:

  1. Drives `docs/_static/code-selector.js` in Node with a DOM stub and
     enumerates every (task x device x direction) combo plus every
     quickstart preset.
  2. Calls `buildDataBlock` + `buildSklearnBlock` on each combo and runs
     `ast.parse` on the generated Python.
  3. Instantiates every extractor (`task.*.stim`, `device.*.neuro`) against
     the installed `neuralset` package so pydantic validates the kwargs.

Run:
    python docs/validate_code_selector.py
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS_FILE = ROOT / "docs" / "_static" / "code-selector.js"


NODE_PRELUDE = r"""
// Minimal DOM stub: enough for code-selector.js to run without a browser.
const fakeEl = {
  value: '',
  innerHTML: '',
  parentElement: { querySelector: () => null, appendChild: () => {} },
  querySelector: () => null,
  addEventListener: () => {},
  classList: { add: () => {}, remove: () => {} },
  appendChild: () => {},
  removeChild: () => {},
  select: () => {},
};
global.document = {
  addEventListener: (event, cb) => { if (event === 'DOMContentLoaded') cb(); },
  getElementById: () => fakeEl,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ ...fakeEl }),
  execCommand: () => {},
  body: { appendChild: () => {}, removeChild: () => {} },
};
"""

NODE_POSTLUDE = r"""
const { task, device, studyMap, presets, buildDataBlock, buildSklearnBlock } = global.__cs;
const fs = require('fs');

const out = { combos: [] };

// Selector combinations (the UI's cartesian product)
for (const [tkey, tsk] of Object.entries(task)) {
  for (const [dkey, dev] of Object.entries(device)) {
    if (dkey === 'fmri_proj') continue; // only reachable from presets
    const study = (studyMap[tkey] || {})[dkey] || 'YourStudy';
    for (const dir of ['decoding', 'encoding']) {
      out.combos.push({
        kind: 'selector', taskKey: tkey, deviceKey: dkey, direction: dir,
        study, installDeps: '',
        dataBlock: buildDataBlock(tsk, dev, study, ''),
        sklearnBlock: buildSklearnBlock(tsk, dev, dir),
      });
    }
  }
}

// Quickstart presets
for (const [pkey, p] of Object.entries(presets)) {
  const tsk = task[p.taskKey];
  const dev = device[p.deviceKey];
  for (const dir of ['decoding', 'encoding']) {
    out.combos.push({
      kind: 'preset', preset: pkey, taskKey: p.taskKey, deviceKey: p.deviceKey,
      direction: dir, study: p.study, installDeps: p.installDeps || '',
      dataBlock: buildDataBlock(tsk, dev, p.study, p.installDeps || ''),
      sklearnBlock: buildSklearnBlock(tsk, dev, dir),
    });
  }
}

// Per-entry extractor strings for individual syntax/instantiation checks
out.taskStims = Object.fromEntries(
  Object.entries(task).map(([k, v]) => [k, v.stim])
);
out.deviceNeuros = Object.fromEntries(
  Object.entries(device).map(([k, v]) => [k, v.neuro])
);

fs.writeFileSync(process.argv[2], JSON.stringify(out));
"""


def render_all_blocks() -> dict:
    """Run the JS in Node to get every generated Python block."""
    src = JS_FILE.read_text()
    # Inject an escape hatch at the very end of the DOMContentLoaded callback so
    # the internal configs and builders leak out to node.
    injected = re.sub(
        r"render\(\);\s*\}\);\s*$",
        "render();\n"
        "  global.__cs = { task, device, studyMap, presets, "
        "buildDataBlock, buildSklearnBlock };\n"
        "});\n",
        src,
    )
    full_js = NODE_PRELUDE + "\n" + injected + "\n" + NODE_POSTLUDE
    driver = "/tmp/cs_driver.js"
    out_path = "/tmp/cs_out.json"
    Path(driver).write_text(full_js)
    Path(out_path).write_text("")
    try:
        result = subprocess.run(
            ["node", driver, out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr[:4000])
            raise RuntimeError(f"node driver exited {result.returncode}")
        raw = Path(out_path).read_text()
        if not raw.strip():
            sys.stderr.write("node stdout:\n" + result.stdout + "\n")
            sys.stderr.write("node stderr:\n" + result.stderr + "\n")
            raise RuntimeError("node driver produced no JSON output")
        return json.loads(raw)
    finally:
        pass


def check_syntax(blocks: list[dict]) -> list[str]:
    errors = []
    for c in blocks:
        for which in ("dataBlock", "sklearnBlock"):
            try:
                ast.parse(c[which])
            except SyntaxError as e:
                label = f"{c['kind']} {c.get('preset') or ''} {c['taskKey']}/{c['deviceKey']}/{c['direction']} :: {which}"
                errors.append(f"SyntaxError in {label}\n  {e}\n---\n{c[which]}\n---")
    return errors


def check_instantiation(task_stims: dict, device_neuros: dict) -> list[str]:
    errors = []
    with tempfile.TemporaryDirectory() as cache:
        import neuralset as ns  # noqa: F401

        infra = {"folder": cache, "cluster": None}
        g = {"ns": ns, "infra": infra}

        for key, src in task_stims.items():
            try:
                exec(src, g)
            except Exception as e:
                errors.append(
                    f"task.{key}.stim failed to instantiate:\n  {e!r}\n  src: {src!r}"
                )

        for key, src in device_neuros.items():
            try:
                exec(src, g)
            except Exception as e:
                errors.append(
                    f"device.{key}.neuro failed to instantiate:\n  {e!r}\n  src: {src!r}"
                )
    return errors


def main() -> int:
    print(f"Driving {JS_FILE.relative_to(ROOT)} via node …")
    data = render_all_blocks()
    print(f"  generated {len(data['combos'])} (task × device × direction) combos")

    syntax_errors = check_syntax(data["combos"])
    if syntax_errors:
        print(f"\n{len(syntax_errors)} SYNTAX errors:")
        for e in syntax_errors:
            print(e)
    else:
        print(f"  ast.parse OK for all {len(data['combos']) * 2} blocks")

    print("\nInstantiating extractors …")
    inst_errors = check_instantiation(data["taskStims"], data["deviceNeuros"])
    if inst_errors:
        print(f"\n{len(inst_errors)} INSTANTIATION errors:")
        for e in inst_errors:
            print(e)
    else:
        print(
            f"  all {len(data['taskStims'])} task stims and "
            f"{len(data['deviceNeuros'])} device neuros instantiated"
        )

    return 0 if not (syntax_errors + inst_errors) else 1


if __name__ == "__main__":
    sys.exit(main())
