/**
 * Fully customizable model-comparison scatter for the NeuralBench landing page.
 *
 * One marker per model.  Independent dropdowns choose which dimension drives
 * the x-axis, y-axis, colour (hue) and marker size, from:
 *   - Year of publication
 *   - Performance (mean rank over the benchmark, or a selected downstream task)
 *   - Number of parameters
 *   - Pretraining corpus size (#subjects / #hours)
 *   - Pretraining objective         (categorical)
 *   - Architecture                  (categorical)
 *   - fMRI space                    (categorical)
 *   - Model family                  (categorical)
 *
 * Pretraining dimensions are only defined for foundation models; any model
 * with a missing value on the chosen x/y dimension is dropped from the plot.
 *
 * Extra controls: log-scale toggles for each axis, and a PNG export button.
 * The legend sits under the figure and is interactive (Plotly click / double
 * click toggles categories on and off).
 *
 * Built on Plotly (loaded from the same CDN as the parallel-coordinates
 * figure).  Data comes from neuralbench-model-scatter-data.json.
 */

(function () {
  "use strict";

  var CONTAINER_ID = "neuralbench-model-scatter";
  var PLOT_ID = "neuralbench-model-scatter-plot";
  var SCRIPT_FILENAME = "neuralbench-model-scatter.js";
  var DATA_FILENAME = "neuralbench-model-scatter-data.json";
  var PLOTLY_SRC = "https://cdn.plot.ly/plotly-3.0.1.min.js";

  // Stable categorical palette (Tableau 20), assigned in category order.
  var PALETTE = [
    "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948",
    "#E15759", "#FF9DA7", "#9C755F", "#86BCB6", "#D37295", "#A0CBE8",
    "#FFBE7D", "#8CD17D", "#B6992D", "#BAB0AC", "#79706E", "#D4A6C8",
  ];

  // Fixed, legible colours for the model-family hue (the default).
  var FAMILY_COLORS = {
    Foundation: "#ff5252",
    "Task-specific": "#448aff",
    Baseline: "#9e9e9e",
  };

  // Sentinel task value: aggregate the whole benchmark (mean rank) instead of
  // a single downstream task.
  var TASK_ALL = "__all__";

  // Available plotting dimensions.  ``needsTask`` dims read model.perf[task].
  var DIMENSIONS = [
    { key: "year", label: "Year", type: "numeric" },
    { key: "performance", label: "Performance", type: "numeric", needsTask: true },
    { key: "n_params", label: "#params", type: "numeric" },
    { key: "pretrain_n_subjects", label: "Pretrain #subjects", type: "numeric" },
    { key: "pretrain_n_hours", label: "Pretrain #hours", type: "numeric" },
    { key: "pretrain_objective", label: "Pretrain objective", type: "categorical" },
    { key: "architecture", label: "Architecture", type: "categorical" },
    { key: "fmri_space", label: "fMRI space", type: "categorical" },
    { key: "family_label", label: "Family", type: "categorical" },
  ];

  var DEFAULT_STATE = {
    x: "year",
    y: "performance",
    hue: "family_label",
    size: "n_params",
    task: TASK_ALL,
    logx: false,
    logy: false,
  };

  var state = {
    data: null,
    x: DEFAULT_STATE.x,
    y: DEFAULT_STATE.y,
    hue: DEFAULT_STATE.hue,
    size: DEFAULT_STATE.size,
    task: DEFAULT_STATE.task,
    logx: DEFAULT_STATE.logx,
    logy: DEFAULT_STATE.logy,
  };

  // ------------------------------------------------------------------
  // Lookups & value access
  // ------------------------------------------------------------------

  function dimByKey(key) {
    for (var i = 0; i < DIMENSIONS.length; i++) {
      if (DIMENSIONS[i].key === key) return DIMENSIONS[i];
    }
    return null;
  }

  /** True when a task-dependent dimension is mapped to any channel. */
  function taskDimActive() {
    var keys = [state.x, state.y, state.hue, state.size];
    for (var i = 0; i < keys.length; i++) {
      var d = dimByKey(keys[i]);
      if (d && d.needsTask) return true;
    }
    return false;
  }

  function taskById(id) {
    var tasks = state.data.tasks;
    for (var i = 0; i < tasks.length; i++) {
      if (tasks[i].id === id) return tasks[i];
    }
    return null;
  }

  /** Raw value of *model* on dimension *dim*; ``null`` when unknown. */
  function dimValue(dim, model) {
    if (!dim) return null;
    if (dim.needsTask) {
      // "All (mean rank)" aggregates the benchmark rather than one task.
      if (state.task === TASK_ALL) {
        return typeof model.mean_rank === "number" ? model.mean_rank : null;
      }
      var v = model.perf ? model.perf[state.task] : undefined;
      return typeof v === "number" ? v : null;
    }
    var raw = model[dim.key];
    return raw === undefined || raw === null ? null : raw;
  }

  /** Whether a dimension currently reads best-is-lowest (reversed axis). */
  function dimBetterLow(dim) {
    if (!dim) return false;
    if (dim.betterLow) return true;
    if (dim.needsTask) {
      // "All (mean rank)" is a rank (lower is better); a single task follows
      // its own metric orientation (e.g. an error/RMSE metric).
      if (state.task === TASK_ALL) return true;
      var t = taskById(state.task);
      return !!(t && t.lower_is_better);
    }
    return false;
  }

  /** Axis / hover label, expanded for the task-dependent performance dim. */
  function dimLabel(dim) {
    if (dim && dim.needsTask) {
      if (state.task === TASK_ALL) return "Mean rank";
      var t = taskById(state.task);
      if (t) return "Performance \u2014 " + t.display + " (" + t.metric_display + ")";
    }
    return dim ? dim.label : "";
  }

  // ------------------------------------------------------------------
  // Formatting & theme
  // ------------------------------------------------------------------

  function fmt(dim, v) {
    if (v === null || v === undefined) return "\u2014";
    if (typeof v !== "number") return String(v);
    if (dim && dim.key === "year") return String(v);
    if (Math.abs(v) >= 1000) {
      return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
    }
    if (Number.isInteger(v)) return String(v);
    return String(Math.round(v * 1000) / 1000);
  }

  function isDarkMode() {
    var body = document.body;
    if (body.dataset.theme === "dark") return true;
    if (body.dataset.theme === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function themeColors() {
    var dark = isDarkMode();
    return {
      font: dark ? "#e6e6e6" : "#2a2a2a",
      grid: dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.10)",
      zero: dark ? "rgba(255,255,255,0.28)" : "rgba(0,0,0,0.22)",
      markerLine: dark ? "rgba(0,0,0,0.55)" : "rgba(255,255,255,0.85)",
    };
  }

  // ------------------------------------------------------------------
  // Colours & sizes
  // ------------------------------------------------------------------

  function orderedCategories(dim, models) {
    var seen = {};
    var out = [];
    for (var i = 0; i < models.length; i++) {
      var v = dimValue(dim, models[i]);
      var label = v === null ? "N/A" : String(v);
      if (!seen[label]) {
        seen[label] = true;
        out.push(label);
      }
    }
    out.sort();
    return out;
  }

  function categoryColor(dim, label, orderedCats) {
    if (label === "N/A") return "#9e9e9e";
    if (dim.key === "family_label" && FAMILY_COLORS[label]) return FAMILY_COLORS[label];
    var idx = orderedCats.indexOf(label);
    return PALETTE[(idx < 0 ? 0 : idx) % PALETTE.length];
  }

  function sizeStats(dim, points) {
    var vals = [];
    for (var i = 0; i < points.length; i++) {
      var v = dimValue(dim, points[i].m);
      if (typeof v === "number") vals.push(v);
    }
    if (!vals.length) return null;
    var lo = Math.min.apply(null, vals);
    var hi = Math.max.apply(null, vals);
    // Log mapping when strictly positive and spanning >2 orders of magnitude
    // (e.g. parameter counts from 1.5K to 157M).
    var useLog = lo > 0 && hi / lo > 100;
    return { min: lo, max: hi, useLog: useLog };
  }

  var SIZE_MIN_PX = 9;
  var SIZE_MAX_PX = 30;

  function markerPx(dim, model, stats) {
    if (!dim || !stats) return 14;
    var v = dimValue(dim, model);
    if (typeof v !== "number") return SIZE_MIN_PX;
    if (stats.max <= stats.min) return 18;
    var t;
    if (stats.useLog) {
      t = (Math.log(v) - Math.log(stats.min)) / (Math.log(stats.max) - Math.log(stats.min));
    } else {
      t = (v - stats.min) / (stats.max - stats.min);
    }
    t = Math.max(0, Math.min(1, t));
    return SIZE_MIN_PX + t * (SIZE_MAX_PX - SIZE_MIN_PX);
  }

  // ------------------------------------------------------------------
  // Trace construction
  // ------------------------------------------------------------------

  function hoverText(model, xDim, yDim, sizeDim, hueDim) {
    var lines = ["<b>" + model.display + "</b>"];
    lines.push(dimLabel(xDim) + ": " + fmt(xDim, dimValue(xDim, model)));
    lines.push(dimLabel(yDim) + ": " + fmt(yDim, dimValue(yDim, model)));
    if (sizeDim && sizeDim.key !== xDim.key && sizeDim.key !== yDim.key) {
      lines.push(dimLabel(sizeDim) + ": " + fmt(sizeDim, dimValue(sizeDim, model)));
    }
    if (
      hueDim &&
      hueDim.key !== xDim.key &&
      hueDim.key !== yDim.key &&
      (!sizeDim || hueDim.key !== sizeDim.key)
    ) {
      lines.push(dimLabel(hueDim) + ": " + fmt(hueDim, dimValue(hueDim, model)));
    }
    return lines.join("<br>");
  }

  function baseMarker(colors, sizes, color) {
    return {
      size: sizes,
      color: color,
      line: { width: 1.2, color: colors.markerLine },
      opacity: 0.92,
    };
  }

  function buildTraces(colors) {
    var xDim = dimByKey(state.x);
    var yDim = dimByKey(state.y);
    var sizeDim = state.size === "none" ? null : dimByKey(state.size);
    var hueDim = state.hue === "none" ? null : dimByKey(state.hue);

    // Keep only models with a defined value on both axes.
    var points = [];
    var models = state.data.models;
    for (var i = 0; i < models.length; i++) {
      var m = models[i];
      if (dimValue(xDim, m) === null || dimValue(yDim, m) === null) continue;
      points.push({ m: m, x: dimValue(xDim, m), y: dimValue(yDim, m) });
    }

    var sStats = sizeDim ? sizeStats(sizeDim, points) : null;

    function pointToTrace(subset, name, color, showLegend) {
      var xs = [], ys = [], texts = [], hover = [], sizes = [];
      for (var j = 0; j < subset.length; j++) {
        var p = subset[j];
        xs.push(p.x);
        ys.push(p.y);
        texts.push(p.m.display);
        hover.push(hoverText(p.m, xDim, yDim, sizeDim, hueDim));
        sizes.push(markerPx(sizeDim, p.m, sStats));
      }
      return {
        type: "scatter",
        mode: "markers+text",
        name: name,
        x: xs,
        y: ys,
        text: texts,
        textposition: "top center",
        textfont: { size: 10, color: colors.font },
        hovertext: hover,
        hovertemplate: "%{hovertext}<extra></extra>",
        marker: baseMarker(colors, sizes, color),
        showlegend: showLegend !== false,
        cliponaxis: false,
      };
    }

    if (hueDim && hueDim.type === "categorical") {
      var cats = orderedCategories(hueDim, points.map(function (p) { return p.m; }));
      var traces = [];
      for (var c = 0; c < cats.length; c++) {
        var label = cats[c];
        var subset = points.filter(function (p) {
          var v = dimValue(hueDim, p.m);
          return (v === null ? "N/A" : String(v)) === label;
        });
        if (!subset.length) continue;
        traces.push(pointToTrace(subset, label, categoryColor(hueDim, label, cats), true));
      }
      return traces;
    }

    if (hueDim && hueDim.type === "numeric") {
      var colorVals = points.map(function (p) {
        var v = dimValue(hueDim, p.m);
        return typeof v === "number" ? v : null;
      });
      var tr = pointToTrace(points, "Models", null, false);
      tr.marker.color = colorVals;
      tr.marker.colorscale = "Viridis";
      tr.marker.showscale = true;
      tr.marker.colorbar = {
        title: { text: dimLabel(hueDim), side: "right" },
        thickness: 12,
        len: 0.7,
        outlinewidth: 0,
        tickfont: { color: colors.font },
        titlefont: { color: colors.font },
      };
      return [tr];
    }

    // No hue: a single uniform-colour trace.
    return [pointToTrace(points, "Models", "#448aff", true)];
  }

  // ------------------------------------------------------------------
  // Layout & drawing
  // ------------------------------------------------------------------

  function axisConfig(dim, isLog, colors) {
    var ax = {
      title: { text: dimLabel(dim), font: { color: colors.font, size: 13 } },
      tickfont: { color: colors.font },
      gridcolor: colors.grid,
      zerolinecolor: colors.zero,
      linecolor: colors.grid,
    };
    if (dim.type === "categorical") {
      ax.type = "category";
      ax.categoryorder = "category ascending";
      return ax;
    }
    if (isLog) ax.type = "log";
    // Put the best (e.g. rank 1) at the top / right. This is independent of the
    // log transform, so it must be applied for log axes too.
    if (dimBetterLow(dim)) ax.autorange = "reversed";
    return ax;
  }

  function draw() {
    var plotEl = document.getElementById(PLOT_ID);
    if (!plotEl || !window.Plotly) return;
    var colors = themeColors();

    var xDim = dimByKey(state.x);
    var yDim = dimByKey(state.y);
    var traces = buildTraces(colors);

    var layout = {
      margin: { l: 72, r: 24, t: 12, b: 120 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: colors.font },
      hovermode: "closest",
      hoverlabel: { align: "left" },
      xaxis: axisConfig(xDim, state.logx, colors),
      yaxis: axisConfig(yDim, state.logy, colors),
      legend: {
        orientation: "h",
        xanchor: "center",
        x: 0.5,
        yanchor: "top",
        y: -0.22,
        font: { color: colors.font },
        title: { text: "" },
      },
      showlegend: true,
    };

    var config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: "neuralbench-model-comparison", scale: 2 },
    };

    window.Plotly.react(plotEl, traces, layout, config);
    updateLogToggleState();
    updateTaskSelectState();
  }

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------

  function optionsFor(dims, includeNone, noneLabel) {
    var s = "";
    if (includeNone) s += '<option value="none">' + noneLabel + "</option>";
    for (var i = 0; i < dims.length; i++) {
      s += '<option value="' + dims[i].key + '">' + escapeHtml(dims[i].label) + "</option>";
    }
    return s;
  }

  function renderControls(container) {
    var numeric = DIMENSIONS.filter(function (d) { return d.type === "numeric"; });

    var taskOpts = '<option value="' + TASK_ALL + '">All (mean rank)</option>';
    var tasks = state.data.tasks;
    for (var i = 0; i < tasks.length; i++) {
      taskOpts +=
        '<option value="' + tasks[i].id + '">' +
        escapeHtml(tasks[i].display) +
        (tasks[i].device && tasks[i].device !== "eeg"
          ? " (" + tasks[i].device.toUpperCase() + ")"
          : "") +
        "</option>";
    }

    var html = "";
    html += '<div class="nbms-controls">';

    html += field("X-axis", '<select id="nbms-x" class="nbms-select">' + optionsFor(DIMENSIONS) + "</select>");
    html += field("Y-axis", '<select id="nbms-y" class="nbms-select">' + optionsFor(DIMENSIONS) + "</select>");
    html += field("Colour", '<select id="nbms-hue" class="nbms-select">' + optionsFor(DIMENSIONS, true, "None") + "</select>");
    html += field("Size", '<select id="nbms-size" class="nbms-select">' + optionsFor(numeric, true, "None (constant)") + "</select>");
    html += field(
      "Task (performance dims)",
      '<select id="nbms-task" class="nbms-select">' + taskOpts + "</select>" +
        '<span id="nbms-task-hint" class="nbms-hint">Set a channel to \u201cPerformance\u201d to use this.</span>',
    );

    html += '<div class="nbms-field nbms-field-actions">';
    html += '<span class="nbms-field-label">Options</span>';
    html += '<div class="nbms-toggle-row">';
    html += '<label class="nbms-check"><input type="checkbox" id="nbms-logx"> log X</label>';
    html += '<label class="nbms-check"><input type="checkbox" id="nbms-logy"> log Y</label>';
    html += '<button type="button" id="nbms-reset" class="nbms-btn nbms-btn-ghost"><i class="fas fa-rotate-left"></i> Reset selection</button>';
    html += '<button type="button" id="nbms-export" class="nbms-btn"><i class="fas fa-download"></i> PNG</button>';
    html += "</div>";
    html += "</div>";

    html += "</div>"; // controls

    html += '<div id="' + PLOT_ID + '" class="nbms-plot"></div>';
    html +=
      '<p class="nbms-note">One marker per model. Pretraining dimensions apply to ' +
      "foundation models only; models without a value on the chosen axis are hidden. " +
      "Click a legend entry to toggle a group; double-click to isolate it.</p>";

    container.innerHTML = html;

    // Reflect current state into the widgets.
    setSelect("nbms-x", state.x);
    setSelect("nbms-y", state.y);
    setSelect("nbms-hue", state.hue);
    setSelect("nbms-size", state.size);
    setSelect("nbms-task", state.task);
    document.getElementById("nbms-logx").checked = state.logx;
    document.getElementById("nbms-logy").checked = state.logy;
  }

  function field(label, control) {
    return (
      '<div class="nbms-field"><span class="nbms-field-label">' +
      escapeHtml(label) +
      "</span>" +
      control +
      "</div>"
    );
  }

  function setSelect(id, value) {
    var el = document.getElementById(id);
    if (el) el.value = value;
  }

  /** Disable log toggles when the corresponding axis is categorical. */
  function updateLogToggleState() {
    var xDim = dimByKey(state.x);
    var yDim = dimByKey(state.y);
    var lx = document.getElementById("nbms-logx");
    var ly = document.getElementById("nbms-logy");
    if (lx) lx.disabled = xDim.type !== "numeric";
    if (ly) ly.disabled = yDim.type !== "numeric";
  }

  /** Grey out the task picker unless a "Task perf." dimension is in use. */
  function updateTaskSelectState() {
    var sel = document.getElementById("nbms-task");
    if (!sel) return;
    var active = taskDimActive();
    sel.disabled = !active;
    var wrapper = sel.parentNode;
    if (wrapper && wrapper.classList) {
      wrapper.classList.toggle("nbms-field-muted", !active);
    }
  }

  function bindEvents() {
    document.getElementById("nbms-x").addEventListener("change", function (e) {
      state.x = e.target.value;
      draw();
    });
    document.getElementById("nbms-y").addEventListener("change", function (e) {
      state.y = e.target.value;
      draw();
    });
    document.getElementById("nbms-hue").addEventListener("change", function (e) {
      state.hue = e.target.value;
      draw();
    });
    document.getElementById("nbms-size").addEventListener("change", function (e) {
      state.size = e.target.value;
      draw();
    });
    document.getElementById("nbms-task").addEventListener("change", function (e) {
      state.task = e.target.value;
      draw();
    });
    document.getElementById("nbms-logx").addEventListener("change", function (e) {
      state.logx = e.target.checked;
      draw();
    });
    document.getElementById("nbms-logy").addEventListener("change", function (e) {
      state.logy = e.target.checked;
      draw();
    });
    document.getElementById("nbms-reset").addEventListener("click", function () {
      state.x = DEFAULT_STATE.x;
      state.y = DEFAULT_STATE.y;
      state.hue = DEFAULT_STATE.hue;
      state.size = DEFAULT_STATE.size;
      state.task = DEFAULT_STATE.task;
      state.logx = DEFAULT_STATE.logx;
      state.logy = DEFAULT_STATE.logy;
      setSelect("nbms-x", state.x);
      setSelect("nbms-y", state.y);
      setSelect("nbms-hue", state.hue);
      setSelect("nbms-size", state.size);
      setSelect("nbms-task", state.task);
      document.getElementById("nbms-logx").checked = state.logx;
      document.getElementById("nbms-logy").checked = state.logy;
      draw();
    });
    document.getElementById("nbms-export").addEventListener("click", function () {
      var plotEl = document.getElementById(PLOT_ID);
      if (!plotEl || !window.Plotly) {
        console.warn("NeuralBench model scatter: Plotly not ready, cannot export.");
        return;
      }
      var rect = plotEl.getBoundingClientRect();
      window.Plotly.downloadImage(plotEl, {
        format: "png",
        filename: "neuralbench-model-comparison",
        scale: 2,
        width: Math.round(rect.width) || 900,
        height: Math.round(rect.height) || 560,
      }).catch(function (err) {
        console.error("NeuralBench model scatter: PNG export failed", err);
      });
    });
  }

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Directory this script was served from (e.g. ".../_static/"). */
  function scriptDir() {
    var scripts = document.getElementsByTagName("script");
    for (var i = 0; i < scripts.length; i++) {
      var src = scripts[i].src || "";
      var idx = src.indexOf(SCRIPT_FILENAME);
      if (idx !== -1) return src.substring(0, idx);
    }
    return "_static/";
  }

  function resolveDataUrl() {
    return scriptDir() + DATA_FILENAME;
  }

  function loadScript(src, onload, onerror) {
    var s = document.createElement("script");
    s.src = src;
    s.charset = "utf-8";
    s.onload = onload;
    s.onerror = onerror;
    document.head.appendChild(s);
    return s;
  }

  // Load Plotly from the CDN if it isn't already on the page (same source as
  // the parallel-coordinates figure).
  function ensurePlotly(cb) {
    if (window.Plotly) {
      cb();
      return;
    }
    var existing = document.getElementById("nbms-plotly-lib");
    if (existing) {
      existing.addEventListener("load", cb);
      return;
    }
    var s = loadScript(PLOTLY_SRC, cb, function () {
      var container = document.getElementById(CONTAINER_ID);
      if (container) {
        container.innerHTML =
          '<p style="color:#888;font-style:italic">Plotly could not be loaded, so the ' +
          "model scatter is unavailable.</p>";
      }
    });
    s.id = "nbms-plotly-lib";
  }

  function observeDarkMode() {
    var observer = new MutationObserver(function () {
      draw();
    });
    observer.observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    }
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------

  function init() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    fetch(resolveDataUrl())
      .then(function (r) { return r.json(); })
      .then(function (data) {
        state.data = data;
        renderControls(container);
        bindEvents();
        observeDarkMode();
        ensurePlotly(draw);
      })
      .catch(function (err) {
        container.innerHTML =
          '<p style="color:#888;font-style:italic">The model scatter could not be loaded.</p>';
        console.error("NeuralBench model scatter: failed to load data", err);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
