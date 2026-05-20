/* code-builder.js — Interactive 6-axis code builder.
   Reads `window.CB_DATA` (generated from docs/_data/code-builder-data.yaml)
   and renders bash + python snippets that wire a study (real public dataset
   when (neuro, stim) maps to one, or `ExampleMultiModal` otherwise) through
   NeuralSet's extractors / Segmenter, and (optionally) an sklearn
   `cross_val_score` one-liner. The yaml-style python embeds the config as a
   triple-quoted string; for encode/decode tasks it wraps the pipeline in an
   `Experiment` model whose `score()` method is cached by `exca.TaskInfra`.

   Architecture
   ────────────
   `createRenderer(DATA, sel)` is the pure renderer — closes over the data
   and the live selection, returns `{buildInstall, buildScriptKwargs,
   buildScriptYaml}`. No DOM access. The browser bootstrap below wires it
   to `<div class="code-builder">`; the Node fixture generator at
   `docs/_data/render_fixtures.mjs` reuses the same factory to freeze
   golden snapshots used by `docs/test_code_builder.py`. */

(function () {
  var AXIS_ORDER = ["neuro", "stim", "task", "model", "compute", "style"];

  // ── Pure renderer factory (no DOM) ────────────────────────────────────────
  function createRenderer(DATA, sel) {
    var AXES = DATA.axes;

    // ── Resolvers ───────────────────────────────────────────────────────────
    function neuro() { return AXES.neuro.options[sel.neuro]; }
    function stim()  { return AXES.stim.options[sel.stim]; }
    function task()  { return AXES.task.options[sel.task]; }
    function comp()  { return AXES.compute.options[sel.compute]; }
    function model() { return AXES.model.options[sel.model]; }
    // Pick a real public dataset when (neuro, stim) maps to one; otherwise
    // fall back to the bundled synthetic study. Looked up via the
    // top-level `studies` map in code-builder-data.yaml.
    function study() {
      var key = sel.neuro + "-" + sel.stim;
      return (DATA.studies && DATA.studies[key]) || DATA.default_study;
    }

    // ── Section: bash install ───────────────────────────────────────────────
    // Quote any package spec carrying `extras` brackets so zsh/bash don't
    // treat the `[...]` as a glob pattern (e.g. `neuralset[tutorials]`).
    function quotePip(p) { return p.indexOf("[") >= 0 ? "'" + p + "'" : p; }
    function uniq(arr) {
      var seen = {}, out = [];
      arr.forEach(function (p) { if (p && !seen[p]) { seen[p] = 1; out.push(p); } });
      return out;
    }
    // Render the install block as two `pip install` calls so framework +
    // extractor deps are visually separated from dataset deps. The dataset
    // section is omitted entirely for ExampleMultiModal (no `pip_packages`).
    function buildInstall() {
      var extras  = uniq([].concat(neuro().pip || [], stim().pip || [], study().pip || []));
      var fwPkgs  = uniq([].concat(neuro().pip_packages || [], stim().pip_packages || []));
      var nsToken = extras.length ? "'neuralset[" + extras.join(",") + "]'" : "neuralset";
      var fwLine  = "pip install " + [nsToken].concat(fwPkgs.map(quotePip)).join(" ");

      var lines = ["# NeuralSet + extractor dependencies", fwLine];
      [neuro().post_install, stim().post_install].forEach(function (p) {
        if (p) lines.push(p);
      });

      var dsPkgs = study().pip_packages || [];
      if (dsPkgs.length) {
        lines.push("");
        lines.push("# Dataset: " + study().name);
        lines.push("pip install " + dsPkgs.map(quotePip).join(" "));
      }
      return lines.join("\n");
    }

    // ── Helpers shared by both rendering styles ─────────────────────────────
    var EVENTS_LINE = "events = study.run()  # simple pd.DataFrame";
    function directionLabel() {
      return task().direction === "decoding"
        ? "Decoding (brain -> stim)"
        : "Encoding (stim -> brain)";
    }
    function loadDemoLines(indent) {
      // Show one batch of 8 segments + the shape of each modality. Use the
      // standard PyTorch DataLoader API so this snippet works inside any
      // existing torch training loop.
      var pad = " ".repeat(indent || 0);
      return [
        pad + "loader = DataLoader(dset, batch_size=8, collate_fn=dset.collate_fn)",
        pad + "batch = next(iter(loader))",
        pad + "print('neuro', batch.data[\"neuro\"].shape)",
        pad + "print('stim', batch.data[\"stim\"].shape)",
      ];
    }
    // Imports needed by the load-only demo (DataLoader). ML tasks usually
    // don't need it, except when the Torch model branch streams batches
    // through a DataLoader — and even then, mlImports() emits the same line
    // already, so we de-dupe by skipping it here in that case.
    function loadImports() {
      if (task().needs_ml) return [];
      return ["from torch.utils.data import DataLoader"];
    }

    // ── Multiline call helper ───────────────────────────────────────────────
    // Given a list of "k=v" fragments and an optional trailing one (e.g.
    // `infra=infra`), produce:
    //
    //     <prefix>(
    //         k1=v1,
    //         k2=v2,
    //         infra=infra,
    //     )
    function multilineCall(prefix, kwargs, trailing) {
      var items = (kwargs || []).slice();
      if (trailing) items.push(trailing);
      if (!items.length) return prefix + "()";
      return prefix + "(\n    "
        + items.join(",\n    ")
        + ",\n)";
    }

    // ── kwargs list → YAML mapping block ────────────────────────────────────
    function pyValueToYaml(v) {
      v = v.trim();
      // Python tuple → YAML flow-list: ("seeg",) -> [seeg]; (0.5, 30) -> [0.5, 30]
      if (v[0] === "(" && v[v.length - 1] === ")") {
        var inner = v.slice(1, -1).replace(/,\s*$/, "").trim();
        var items = inner.split(",").map(function (s) {
          var t = s.trim();
          if (/^['"].*['"]$/.test(t)) t = t.slice(1, -1);
          return t;
        });
        return "[" + items.join(", ") + "]";
      }
      if (/^["'].*["']$/.test(v) && v.indexOf("\\") === -1) return v.slice(1, -1);
      return v;
    }
    function kwargsListToYamlBlock(kwList, indentSpaces) {
      var pad = " ".repeat(indentSpaces);
      return (kwList || []).map(function (kv) {
        var eq = kv.indexOf("=");
        var k = kv.slice(0, eq).trim();
        var v = pyValueToYaml(kv.slice(eq + 1));
        return pad + k + ": " + v;
      }).join("\n");
    }

    // ── ML body ─────────────────────────────────────────────────────────────
    // Two flavours, switched by `model()`:
    //   * ridge → classic full-RAM `dset.load_all()` + cross_val_score (the
    //             "fits in memory" sklearn one-liner).
    //   * torch → mini-batch training over a DataLoader with manual
    //             zero_grad / loss / step loop (showcases that a NeuralSet
    //             dataset plugs straight into the standard PyTorch
    //             streaming pattern).
    // mlImports / mlLabel / mlMetricExpr / mlComputeLines all dispatch on
    // the model kind. `mlComputeLines` returns *body* lines only; the
    // caller emits the final `print(...)` (or `return float(...)`) using
    // mlLabel / mlMetricExpr.

    // Common machinery: assemble the X / y expressions used by both flavours
    // (so the sklearn path stays a one-liner and the torch path can reuse the
    // same expressions inside the training loop).
    function _mlExprs() {
      var s = stim(), n = neuro(), t = task();
      var isClass = !!s.is_classification;
      var isDec = (t.direction === "decoding");
      var tLines = [];
      var xNeuro;
      if (n.x_expr) {
        xNeuro = n.x_expr;
      } else {
        var tCmt = n.t_comment ? "  # " + n.t_comment : "";
        tLines.push("t = " + n.t + tCmt);
        xNeuro = 'batch.data["neuro"][:, :, t]';
      }
      // Stim flattened to 2D for sklearn / torch regression heads, or argmax'd
      // to long indices for cross-entropy classification heads.
      var stimFlat   = 'batch.data["stim"].reshape(len(batch), -1)';
      var stimLabels = 'batch.data["stim"].argmax(-1)';
      var stimOneHot = 'batch.data["stim"].float()'; // for encoding from class
      var yStim = isClass ? stimLabels : stimFlat;
      // For encoding, the input is the stimulus. When stim is a one-hot
      // classification target we cast to float so nn.Linear / sklearn accept
      // it; for embeddings we keep the flat 2D form.
      var xStim = isClass ? stimOneHot : stimFlat;
      return {
        isClass: isClass, isDec: isDec, tLines: tLines,
        xNeuro: xNeuro, yStim: yStim, xStim: xStim,
      };
    }

    function mlImports() {
      var t = task();
      if (!t.needs_ml) return [];
      var m = model();
      if (m.kind === "torch") {
        return [
          "import torch.nn.functional as F",
          "from torch import optim, nn",
          "from torch.utils.data import DataLoader",
        ];
      }
      // ridge: classic full-RAM `dset.load_all()` + `cross_val_score`.
      var e = _mlExprs();
      return [
        "from sklearn.linear_model import "
          + ((e.isDec && e.isClass) ? "RidgeClassifier" : "Ridge"),
        "from sklearn.model_selection import cross_val_score",
      ];
    }

    function mlLabel() {
      var t = task();
      if (!t.needs_ml) return "score";
      var m = model();
      // The torch branch reports a per-batch training loss; only the
      // Ridge / cross_val_score branch yields a "score" metric.
      if (m.kind === "torch") return "final loss";
      var s = stim();
      var isClass = !!s.is_classification;
      var isDec = (t.direction === "decoding");
      return (isDec && isClass) ? "balanced accuracy" : "score";
    }

    // The Python expression printed in the f-string / returned by
    // Experiment.score(). The torch branch exposes `loss` from inside the
    // training loop (a 0-d tensor — works in both `f"{...:.3f}"` and
    // `float(...)`); the Ridge branch yields a CV `scores` ndarray.
    function mlMetricExpr() {
      return model().kind === "ridge" ? "scores.mean()" : "loss";
    }

    function mlComputeLines(indent) {
      var t = task();
      if (!t.needs_ml) return [];
      var pad = " ".repeat(indent);
      var e = _mlExprs();
      var X = e.isDec ? e.xNeuro : e.xStim;
      var y = e.isDec ? e.yStim  : e.xNeuro;
      var m = model();

      if (m.kind === "torch") {
        // For decoding-classification the model outputs `n_classes` logits and
        // we use cross_entropy on long target indices. Every other case is a
        // plain multi-output linear regression with MSE.
        var lossFn, outDimExpr;
        if (e.isDec && e.isClass) {
          lossFn = "F.cross_entropy";
          // `n_classes` from the one-hot last dim — independent of `y` (which
          // is 1D after argmax and would size the head incorrectly).
          outDimExpr = 'batch.data["stim"].shape[-1]';
        } else {
          lossFn = "F.mse_loss";
          outDimExpr = y + ".shape[-1]";
        }
        var lines = [];
        lines.push(pad + "loader = DataLoader(dset, batch_size=32, collate_fn=dset.collate_fn, shuffle=True)");
        e.tLines.forEach(function (l) { lines.push(pad + l); });
        // Peek one batch only to size the Linear head; the actual training
        // pass starts fresh from `loader`.
        lines.push(pad + "batch = next(iter(loader))");
        lines.push(pad + "X = " + X);
        lines.push(pad + "model = nn.Linear(X.shape[-1], " + outDimExpr + ")");
        lines.push(pad + "opt   = optim.Adam(model.parameters(), lr=1e-3)");
        lines.push(pad + "for batch in loader:");
        lines.push(pad + "    X = " + X);
        lines.push(pad + "    y = " + y);
        lines.push(pad + "    opt.zero_grad()");
        lines.push(pad + "    loss = " + lossFn + "(model(X), y)");
        lines.push(pad + "    loss.backward()");
        lines.push(pad + "    opt.step()");
        return lines;
      }

      // ridge — classic full-RAM cross-validated score.
      var est = (e.isDec && e.isClass) ? "RidgeClassifier()" : "Ridge()";
      var lines = [pad + "batch = dset.load_all()"];
      e.tLines.forEach(function (l) { lines.push(pad + l); });
      lines.push(pad + "scores = cross_val_score(");
      lines.push(pad + "    estimator=" + est + ",");
      lines.push(pad + "    X=" + X + ",");
      lines.push(pad + "    y=" + y + ",");
      lines.push(pad + ")");
      return lines;
    }

    // ── infra_literal helpers (shared by kwargs + yaml renderers) ──────────
    // The YAML stores `infra_literal` as a JSON object string (e.g.
    // `'{"folder": "$CACHE", "cluster": "slurm"}'`) — parse it once and
    // emit it as either a Python dict literal (kwargs) or YAML key/value
    // lines (yaml). Anything starting with `$CACHE` is rewritten on the
    // Python side to interpolate the runtime `CACHE` Path.
    function infraToPyDict(infraLiteral) {
      var d = JSON.parse(infraLiteral);
      var parts = Object.keys(d).map(function (k) {
        var v = d[k];
        var rhs;
        // `$CACHE` / `$CACHE/sub` placeholders are interpolated to the
        // runtime `CACHE` Path. Pydantic coerces Path -> str so we don't
        // wrap in `str(...)`.
        if (typeof v === "string" && v === "$CACHE") rhs = "CACHE";
        else if (typeof v === "string" && v.indexOf("$CACHE/") === 0)
          rhs = 'CACHE / "' + v.slice(7) + '"';
        else if (typeof v === "string") rhs = JSON.stringify(v);
        else if (v === null) rhs = "None";
        else rhs = JSON.stringify(v);
        return JSON.stringify(k) + ": " + rhs;
      });
      // Break multi-key dicts (e.g. slurm) onto separate lines for
      // readability; single-key dicts (local cache) stay on one line.
      if (parts.length <= 1) return "{" + parts.join(", ") + "}";
      return "{\n    " + parts.join(",\n    ") + ",\n}";
    }
    function infraToYamlBlock(infraLiteral, indentSpaces) {
      var pad = " ".repeat(indentSpaces);
      var d = JSON.parse(infraLiteral);
      return Object.keys(d).map(function (k) {
        var v = d[k];
        return pad + k + ": " + (v === null ? "null" : v);
      }).join("\n");
    }

    // ── kwargs-style python script ──────────────────────────────────────────
    function buildScriptKwargs() {
      var n = neuro(), s = stim(), c = comp(), t = task();
      var win = n.window;
      // Slurm scripts get an `infra_slurm = {...}` variable so the cluster
      // config is self-documenting at a glance; local scripts use the
      // shorter `infra = {...}`. Both are consumed identically downstream.
      var infraVar = sel.compute === "slurm" ? "infra_slurm" : "infra";
      var sTrailing = s.accepts_infra === false ? null : "infra=" + infraVar;
      var cInfra = infraToPyDict(c.infra_literal);

      var lines = [
        "from pathlib import Path",
        "import neuralset as ns",
      ];
      var mlImps = mlImports();
      mlImps.forEach(function (l) { lines.push(l); });
      loadImports().forEach(function (l) { lines.push(l); });
      lines.push("");
      lines.push('CACHE = Path.home() / "neuroai_data" / ".cache"');
      lines.push('STUDIES = Path.home() / "neuroai_data" / "studies"');
      lines.push("STUDIES.mkdir(parents=True, exist_ok=True)");
      lines.push(infraVar + " = " + cInfra);
      lines.push("");
      var stu = study();
      lines.push("# 1. " + stu.comment);
      // Pass the *parent* studies dir; Study.download() resolves the
      // study-name subfolder. The Study's infra caches the merged events
      // DataFrame and propagates the folder to infra_timelines so each
      // timeline's events are also cached. In local mode we reuse the
      // single `infra` variable (same `{"folder": CACHE}`); in slurm mode
      // we pin the Study to local caching to avoid spawning a Slurm job
      // just to merge the events DataFrame. (YAML mode keeps the
      // explicit subfolder because the Experiment freezes the Study —
      // see buildScriptYaml.)
      var studyInfraExpr = sel.compute === "slurm"
        ? '{"folder": CACHE}'
        : infraVar;
      lines.push("study = ns.Study(");
      lines.push('    name="' + stu.name + '",');
      lines.push("    path=STUDIES,");
      lines.push("    infra=" + studyInfraExpr + ",");
      lines.push(")");
      lines.push("");
      lines.push("# 2. Define extractors");
      // Real-data studies may pin extractor kwargs (e.g. allow_maxshield=True
      // for Bel's MaxShield-recorded MEG). Append them to the neuro/stim
      // kwargs so the rendered call carries them.
      var nKwargs = (n.kwargs || []).concat(stu.neuro_kwargs || []);
      var sKwargs = (s.kwargs || []).concat(stu.stim_kwargs || []);
      lines.push(multilineCall("neuro = ns.extractors." + n.cls, nKwargs, "infra=" + infraVar));
      lines.push(multilineCall("stim  = ns.extractors." + s.cls, sKwargs, sTrailing));
      lines.push("");
      lines.push("# 3. Segment around each \"" + s.event_type + "\" event");
      lines.push("segmenter = ns.Segmenter(");
      lines.push("    start=" + win.start + ", duration=" + win.duration + ",");
      lines.push("    trigger_query='type==\"" + s.event_type + "\"',");
      lines.push("    extractors=dict(neuro=neuro, stim=stim),");
      lines.push("    drop_incomplete=True,");
      lines.push(")");
      lines.push("");
      // All instances defined — now run the pipeline.
      lines.push("# 4. Run the study and apply the segmenter");
      lines.push("study.download()");
      lines.push(EVENTS_LINE);
      lines.push("dset = segmenter.apply(events)");
      lines.push("dset.prepare()");

      var ml = mlComputeLines(0);
      if (ml.length) {
        lines.push("");
        lines.push("# 5. " + directionLabel());
        ml.forEach(function (l) { lines.push(l); });
        lines.push('print(f"' + mlLabel() + ' = {' + mlMetricExpr() + ':.3f}")');
      } else {
        lines.push("");
        lines.push("# 5. Inspect one batch of 8 segments");
        loadDemoLines(0).forEach(function (l) { lines.push(l); });
      }
      return lines.join("\n");
    }

    // ── yaml-style — YAML embedded inline as a Python string ────────────────
    function buildScriptYaml() {
      var n = neuro(), s = stim(), c = comp(), t = task();
      var win = n.window;
      var needsML = !!t.needs_ml;

      // Build the YAML body (inserted into a triple-quoted Python string).
      // Both renderers expose the same kwargs so the kwargs-vs-YAML choice
      // is purely a syntax/ergonomics decision, not a feature comparison.
      var infraExtractor = infraToYamlBlock(c.infra_literal, 8);
      // Experiment-level infra block (TaskInfra) — same literal as the
      // extractor MapInfras so cache + slurm flip atomically. Only emitted
      // when the script defines an Experiment that wraps `score()` with
      // `@infra.apply`.
      var expInfraYaml = needsML
        ? "infra:\n" + infraToYamlBlock(c.infra_literal, 2)
        : null;

      var stu = study();
      var nKw = (n.kwargs || []).concat(stu.neuro_kwargs || []);
      var sKw = (s.kwargs || []).concat(stu.stim_kwargs || []);

      var stimBlock = [
        "    stim:",
        "      name: " + s.cls,
        kwargsListToYamlBlock(sKw, 6),
      ];
      if (s.accepts_infra !== false) {
        stimBlock.push("      infra:");
        stimBlock.push(infraExtractor);
      }

      // `infra: {folder: $CACHE}` caches the merged events DataFrame
      // and propagates the folder to infra_timelines so each timeline
      // is also cached. Same `$CACHE` placeholder as the extractor
      // MapInfras.
      var yamlSections = [
        "# " + stu.comment,
        "study:",
        "  name: " + stu.name,
        "  path: $STUDIES/" + stu.name,
        "  infra:",
        "    folder: $CACHE",
        "segmenter:",
        "  start: " + win.start,
        "  duration: " + win.duration,
        "  trigger_query: 'type==\"" + s.event_type + "\"'",
        "  drop_incomplete: true",
        "  extractors:",
        "    neuro:",
        "      name: " + n.cls,
        kwargsListToYamlBlock(nKw, 6),
        "      infra:",
        infraExtractor,
      ].concat(stimBlock);
      if (expInfraYaml) yamlSections.push(expInfraYaml);
      var yamlBody = yamlSections.join("\n");

      var imports = ["from pathlib import Path", "import yaml, pydantic", "import neuralset as ns"];
      if (needsML) imports.push("import exca");
      mlImports().forEach(function (l) { imports.push(l); });
      loadImports().forEach(function (l) { imports.push(l); });

      var pyLines = imports.slice();
      pyLines.push("");
      pyLines.push("config = '''");
      pyLines.push(yamlBody);
      pyLines.push("'''");
      pyLines.push("");
      pyLines.push('CACHE = Path.home() / "neuroai_data" / ".cache"');
      pyLines.push('STUDIES = Path.home() / "neuroai_data" / "studies"');
      pyLines.push("STUDIES.mkdir(parents=True, exist_ok=True)");
      // `str.replace` requires str args, so str() is unavoidable here
      // (unlike the kwargs-mode infra dict where Pydantic coerces Path).
      // The `config` name is reused: the raw YAML string is replaced by
      // the parsed dict, so downstream code only sees a single `config`.
      pyLines.push("config = yaml.safe_load(");
      pyLines.push('    config.replace("$CACHE", str(CACHE))');
      pyLines.push('          .replace("$STUDIES", str(STUDIES))');
      pyLines.push(")");
      pyLines.push("");

      pyLines.push("class Experiment(pydantic.BaseModel):");
      pyLines.push("    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)");
      pyLines.push("    study: ns.Study");
      pyLines.push("    segmenter: ns.Segmenter");
      if (needsML) {
        pyLines.push("    infra: exca.TaskInfra = exca.TaskInfra()");
        pyLines.push("");
        pyLines.push("    @infra.apply");
        pyLines.push("    def score(self) -> float:");
        pyLines.push("        self.study.download()");
        pyLines.push("        " + EVENTS_LINE.replace("study.run()", "self.study.run()"));
        pyLines.push("        dset = self.segmenter.apply(events)");
        pyLines.push("        dset.prepare()");
        mlComputeLines(8).forEach(function (l) { pyLines.push(l); });
        pyLines.push("        return float(" + mlMetricExpr() + ")");
      }
      pyLines.push("");
      pyLines.push("exp = Experiment(**config)");
      if (needsML) {
        pyLines.push("");
        pyLines.push("# " + directionLabel());
        pyLines.push("score = exp.score()");
        pyLines.push('print(f"' + mlLabel() + ' = {score:.3f}")');
      } else {
        pyLines.push("exp.study.download()");
        pyLines.push(EVENTS_LINE.replace("study.run()", "exp.study.run()"));
        pyLines.push("dset = exp.segmenter.apply(events)");
        pyLines.push("dset.prepare()");
        pyLines.push("");
        pyLines.push("# Inspect one batch of 8 segments");
        loadDemoLines(0).forEach(function (l) { pyLines.push(l); });
      }
      return pyLines.join("\n");
    }

    return {
      buildInstall: buildInstall,
      buildScriptKwargs: buildScriptKwargs,
      buildScriptYaml: buildScriptYaml,
    };
  }

  // ── Node export (CommonJS-only; browser falls through) ─────────────────────
  // The fixture generator (docs/_data/render_fixtures.mjs) imports this as a
  // CJS module and only ever calls `createRenderer`. Skip the DOM bootstrap.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createRenderer: createRenderer, AXIS_ORDER: AXIS_ORDER };
    return;
  }

  // ── Browser bootstrap ─────────────────────────────────────────────────────
  if (typeof window === "undefined" || !window.document) return;

  function init() {
    var root = document.querySelector(".code-builder");
    if (!root || !window.CB_DATA) return;

    var DATA = window.CB_DATA;
    var AXES = DATA.axes;

    // Selection starts at each axis's default — page is fully populated
    // on load (no progressive reveal needed). Any axis can also be seeded
    // from a URL query param (e.g. `?neuro=meg&stim=audio&task=decoding`)
    // so deep-links from the NeuralSet quickstart pills land on a fully
    // configured Code Builder. Unknown axes / option keys are ignored
    // silently — robust to stale links if the YAML evolves.
    var sel = {};
    AXIS_ORDER.forEach(function (a) { sel[a] = AXES[a].default; });
    var params = new URLSearchParams(window.location.search);
    AXIS_ORDER.forEach(function (a) {
      var v = params.get(a);
      if (v && AXES[a].options[v]) sel[a] = v;
    });

    var R = createRenderer(DATA, sel);

    // ── DOM helpers ─────────────────────────────────────────────────────────
    function el(tag, cls, html) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (html != null) e.innerHTML = html;
      return e;
    }
    var H = window.codeHighlight;
    var escapeHtml = H.escapeHtml;
    var highlightPython = H.python;
    var highlightBash = H.bash;

    function render() {
      var bashEl = root.querySelector("#cb-install code");
      if (bashEl) {
        var bash = R.buildInstall();
        bashEl.innerHTML = highlightBash(bash);
        addCopy(bashEl, bash);
      }

      // The Model axis only matters when a task actually trains something —
      // hide it when the user picks "Load data" so the bar stays uncluttered.
      // We toggle inline `display` rather than the `hidden` attribute because
      // `.code-builder .cb-axis { display: flex }` would otherwise win on
      // specificity and keep it visible.
      var modelWrap = root.querySelector('.cb-axis[data-axis="model"]');
      if (modelWrap) modelWrap.style.display = AXES.task.options[sel.task].needs_ml ? "" : "none";

      var py = sel.style === "yaml" ? R.buildScriptYaml() : R.buildScriptKwargs();
      var pyEl = root.querySelector("#cb-script code");
      pyEl.innerHTML = highlightPython(py);
      addCopy(pyEl, py);
    }

    // ── Copy button ─────────────────────────────────────────────────────────
    // Match sphinx-copybutton's markup so the site-wide `button.copybtn`
    // styles (and the `.success` override) apply for free, instead of
    // shipping a one-off Font Awesome icon that drifts from the rest of
    // the documentation.
    var ICON_COPY = '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><title>Copy to clipboard</title><path stroke="none" d="M0 0h24v24H0z" fill="none"/><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"/></svg>';
    var ICON_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-check" width="44" height="44" viewBox="0 0 24 24" stroke-width="2" stroke="#22863a" fill="none" stroke-linecap="round" stroke-linejoin="round"><title>Copied!</title><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M5 12l5 5l10 -10"/></svg>';

    function addCopy(target, content) {
      var wrap = target.parentElement;
      var existing = wrap.querySelector("button.copybtn");
      if (existing) existing.remove();
      var b = el("button", "copybtn o-tooltip--left", ICON_COPY);
      b.type = "button";
      b.setAttribute("data-tooltip", "Copy");
      b.setAttribute("aria-label", "Copy to clipboard");
      b.addEventListener("click", function () {
        var done = function () {
          b.classList.add("success");
          b.innerHTML = ICON_CHECK;
          setTimeout(function () { b.classList.remove("success"); }, 1500);
          setTimeout(function () { b.innerHTML = ICON_COPY; }, 2000);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(content).then(done, done);
        } else {
          var ta = document.createElement("textarea");
          ta.value = content; document.body.appendChild(ta);
          ta.select(); document.execCommand("copy");
          document.body.removeChild(ta);
          done();
        }
      });
      wrap.appendChild(b);
    }

    // ── UI: collapsible axes ────────────────────────────────────────────────
    // Each axis renders as:
    //   ┌──────────────────────────────────┐
    //   │ Axis · <SelectedLabel>     ▾    │   ← always visible (cb-axis-summary)
    //   ├──────────────────────────────────┤
    //   │ [opt1] [opt2] [opt3] …           │   ← expanded only when .cb-open
    //   └──────────────────────────────────┘
    // Clicking the summary toggles the tray; clicking a pill selects it
    // and collapses the tray.
    var bar = root.querySelector(".cb-axes");

    AXIS_ORDER.forEach(function (axis) {
      var meta = AXES[axis];
      var wrap = el("div", "cb-axis");
      wrap.dataset.axis = axis;

      var summary = el("button", "cb-axis-summary");
      summary.type = "button";
      summary.setAttribute("aria-expanded", "false");
      var summaryValue = el("span", "cb-axis-summary-value");
      summary.innerHTML =
        '<span class="cb-axis-label">' + escapeHtml(meta.label) + '</span>';
      summary.appendChild(summaryValue);
      summary.insertAdjacentHTML(
        "beforeend",
        '<span class="cb-axis-caret" aria-hidden="true">▾</span>'
      );
      wrap.appendChild(summary);

      var tray = el("div", "cb-axis-tray");
      var pillByKey = {};
      Object.keys(meta.options).forEach(function (key) {
        var opt = meta.options[key];
        var pill = el("button", "cb-pill", escapeHtml(opt.label));
        pill.type = "button";
        pill.dataset.value = key;
        if (sel[axis] === key) pill.classList.add("active");
        pill.addEventListener("click", function (ev) {
          ev.stopPropagation();
          sel[axis] = key;
          Object.keys(pillByKey).forEach(function (k) {
            pillByKey[k].classList.toggle("active", k === key);
          });
          summaryValue.textContent = opt.label;
          closeTray();
          render();
        });
        pillByKey[key] = pill;
        tray.appendChild(pill);
      });
      wrap.appendChild(tray);

      function openTray() {
        wrap.classList.add("cb-open");
        summary.setAttribute("aria-expanded", "true");
      }
      function closeTray() {
        wrap.classList.remove("cb-open");
        summary.setAttribute("aria-expanded", "false");
      }
      summary.addEventListener("click", function (ev) {
        ev.stopPropagation();
        if (wrap.classList.contains("cb-open")) {
          closeTray();
        } else {
          // Close any other open axis (one-at-a-time disclosure).
          bar.querySelectorAll(".cb-axis.cb-open").forEach(function (a) {
            a.classList.remove("cb-open");
            var s = a.querySelector(".cb-axis-summary");
            if (s) s.setAttribute("aria-expanded", "false");
          });
          openTray();
        }
      });

      // Seed the summary text with the default selection's label.
      summaryValue.textContent = meta.options[sel[axis]].label;

      bar.appendChild(wrap);
    });

    // Click outside any axis → close everything.
    document.addEventListener("click", function (ev) {
      if (!bar.contains(ev.target)) {
        bar.querySelectorAll(".cb-axis.cb-open").forEach(function (a) {
          a.classList.remove("cb-open");
          var s = a.querySelector(".cb-axis-summary");
          if (s) s.setAttribute("aria-expanded", "false");
        });
      }
    });

    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
