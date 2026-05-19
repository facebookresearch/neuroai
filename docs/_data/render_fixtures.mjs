/* render_fixtures.mjs — Render Code Builder fixtures into a target dir.
 *
 * Reuses `createRenderer` from docs/_static/code-builder.js (the same
 * factory the browser bootstrap calls) so the JS renderer is the single
 * source of truth. For each axis-pinned combo (one combo per axis option
 * with the other axes at their defaults) we emit:
 *
 *   <out>/<id>/install.sh
 *   <out>/<id>/script.py
 *
 * Output dir resolution (first match wins):
 *   1. argv[2]                        — `node render_fixtures.mjs /tmp/foo`
 *   2. $CB_FIXTURES_DIR               — env-var override
 *   3. docs/_data/code_builder_fixtures (legacy default; not committed)
 *
 * `docs/test_code_builder.py` calls this script once per session into
 * `tmp_path_factory.mktemp(...)` and consumes the output from there, so
 * fixtures are never committed and can never drift from the renderer.
 */

import { readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const HERE = dirname(fileURLToPath(import.meta.url));
const DOCS = resolve(HERE, "..");
const FIXTURES_DIR = process.argv[2]
  ? resolve(process.argv[2])
  : process.env.CB_FIXTURES_DIR
    ? resolve(process.env.CB_FIXTURES_DIR)
    : resolve(HERE, "code_builder_fixtures");

const require = createRequire(import.meta.url);
const { createRenderer, AXIS_ORDER } = require(
  resolve(DOCS, "_static", "code-builder.js"),
);

// `code-builder-data.js` is `window.CB_DATA = {...};`. Strip the prefix
// and parse the trailing JSON object. We don't `require()` it because
// it isn't valid CJS (assigns to `window`), and we don't want to depend
// on jsdom for a one-line strip.
function loadData() {
  const src = readFileSync(
    resolve(DOCS, "_static", "code-builder-data.js"),
    "utf8",
  );
  const start = src.indexOf("{");
  const end = src.lastIndexOf("}");
  if (start < 0 || end < 0) throw new Error("CB_DATA payload not found");
  return JSON.parse(src.slice(start, end + 1));
}

// One combo per axis option, with the other axes at their defaults.
// Mirrors `_axis_pinned()` in test_code_builder.py.
function axisPinned(DATA) {
  const base = Object.fromEntries(
    AXIS_ORDER.map((a) => [a, DATA.axes[a].default]),
  );
  const seen = new Set();
  const out = [];
  for (const axis of AXIS_ORDER) {
    for (const key of Object.keys(DATA.axes[axis].options)) {
      const sel = { ...base, [axis]: key };
      const sig = AXIS_ORDER.map((a) => sel[a]).join("|");
      if (seen.has(sig)) continue;
      seen.add(sig);
      out.push(sel);
    }
  }
  return out;
}

function selId(sel) {
  return AXIS_ORDER.map((a) => sel[a]).join("-");
}

function renderOne(DATA, sel) {
  const R = createRenderer(DATA, sel);
  return {
    install: R.buildInstall(),
    script: sel.style === "yaml" ? R.buildScriptYaml() : R.buildScriptKwargs(),
  };
}

function writeFixtures() {
  const DATA = loadData();
  const combos = axisPinned(DATA);

  // Wipe and recreate the fixtures dir so stale combos don't linger.
  rmSync(FIXTURES_DIR, { recursive: true, force: true });
  mkdirSync(FIXTURES_DIR, { recursive: true });

  for (const sel of combos) {
    const id = selId(sel);
    const dir = resolve(FIXTURES_DIR, id);
    mkdirSync(dir, { recursive: true });
    const { install, script } = renderOne(DATA, sel);
    writeFileSync(resolve(dir, "install.sh"), install + "\n");
    writeFileSync(resolve(dir, "script.py"), script + "\n");
  }

  const ids = readdirSync(FIXTURES_DIR).sort();
  console.log(`Wrote ${ids.length} fixtures to ${FIXTURES_DIR}`);
  for (const id of ids) console.log("  - " + id);
}

writeFixtures();
