/**
 * Per-category radar plots for the NeuralBench landing page.
 *
 * Renders one radar chart per task category (Cognitive, BCI, Clinical, ...).
 * Each spoke is a task in that category and the radius encodes a model's
 * performance on that task.  Controls allow:
 *   - toggling between normalised score and rank,
 *   - toggling individual models on/off,
 *   - overlaying two aggregate lines ("Task-specific" and "Foundation") that
 *     average the *enabled* members of each family, so they follow the model
 *     selection above (off by default).
 *
 * Data is shared with the results table: it is fetched from
 * neuralbench-results-data.json (see generate_results_json.py).  Each task
 * carries a "category" field and the payload lists the ordered "categories".
 */

(function () {
  "use strict";

  var CONTAINER_ID = "neuralbench-category-radars";
  var SCRIPT_FILENAME = "neuralbench-category-radars.js";
  var DATA_FILENAME = "neuralbench-results-data.json";

  // Aggregate-line colours (family averages).  Chosen to stand apart from the
  // per-model palette and to read clearly in both light and dark themes.
  var AGG_COLORS = {
    classic: "#448aff", // Task-specific average
    foundation: "#ff5252", // Foundation average
  };
  var AGG_LABELS = { classic: "Task-specific", foundation: "Foundation" };

  // Stable categorical palette assigned to models in payload order.
  var PALETTE = [
    "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948",
    "#E15759", "#FF9DA7", "#9C755F", "#86BCB6", "#D37295", "#A0CBE8",
    "#FFBE7D", "#8CD17D", "#B6992D", "#BAB0AC", "#79706E", "#D4A6C8",
  ];

  var state = {
    data: null,
    groups: null, // [{title, categories:[...]}]
    view: "score", // "score" | "rank"
    enabledModels: null, // Set of model names
    aggEnabled: { classic: false, foundation: false },
    modelColor: {}, // name -> colour
    taskStats: {}, // task_id -> {vmin, vmax, maxRank}
  };

  // ------------------------------------------------------------------
  // Theme helpers
  // ------------------------------------------------------------------

  function isDarkMode() {
    var body = document.body;
    if (body.dataset.theme === "dark") return true;
    if (body.dataset.theme === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function themeColors() {
    var dark = isDarkMode();
    return {
      grid: dark ? "rgba(255,255,255,0.14)" : "rgba(0,0,0,0.12)",
      axis: dark ? "rgba(255,255,255,0.22)" : "rgba(0,0,0,0.18)",
      label: dark ? "rgba(230,230,230,0.85)" : "rgba(40,40,40,0.85)",
      cardBorder: dark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.08)",
    };
  }

  // ------------------------------------------------------------------
  // Value computation
  // ------------------------------------------------------------------

  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  // Baselines used to anchor the 0 point of the normalised score, in
  // preference order.  ``dummy`` (majority-class / mean predictor) is the
  // primary floor; ``chance`` is the fallback when a task has no dummy.
  var FLOOR_MODELS = ["dummy", "chance"];

  function taskStats(task) {
    if (state.taskStats[task.id]) return state.taskStats[task.id];
    var row = state.data.results[task.id] || {};
    var means = [];
    var ranks = [];
    for (var name in row) {
      if (!row.hasOwnProperty(name)) continue;
      var e = row[name];
      if (typeof e.mean === "number") means.push(e.mean);
      if (typeof e.rank === "number") ranks.push(e.rank);
    }
    var lower = task.lower_is_better === true;

    // Best achieved score (max, or min for lower-is-better metrics).
    var best = means.length
      ? (lower ? Math.min.apply(null, means) : Math.max.apply(null, means))
      : (lower ? 0 : 1);

    // Floor = the dummy/chance baseline score if available, else the worst
    // observed score.  This is what maps to radius 0.
    var floor = null;
    for (var fi = 0; fi < FLOOR_MODELS.length; fi++) {
      var fe = row[FLOOR_MODELS[fi]];
      if (fe && typeof fe.mean === "number") {
        floor = fe.mean;
        break;
      }
    }
    if (floor === null && means.length) {
      floor = lower ? Math.max.apply(null, means) : Math.min.apply(null, means);
    }

    var stats = {
      best: best,
      floor: floor === null ? (lower ? 1 : 0) : floor,
      lower: lower,
      maxRank: ranks.length ? Math.max.apply(null, ranks) : 1,
    };
    state.taskStats[task.id] = stats;
    return stats;
  }

  /**
   * Normalised radius in [0, 1] for a model on a task (1 = best).
   * Score view: 0 = the dummy/chance baseline, 1 = the best model on that
   * task (so ``dummy`` sits at the centre and anything worse clamps to 0).
   * Rank view: rank 1 -> 1, worst rank -> 0.
   * Returns null when the model has no result for the task.
   */
  function normValue(task, modelName) {
    var entry = (state.data.results[task.id] || {})[modelName];
    if (!entry) return null;
    var stats = taskStats(task);
    if (state.view === "rank") {
      if (typeof entry.rank !== "number") return null;
      if (stats.maxRank <= 1) return 1;
      return clamp01((stats.maxRank - entry.rank) / (stats.maxRank - 1));
    }
    if (typeof entry.mean !== "number") return null;
    var span = stats.lower ? stats.floor - stats.best : stats.best - stats.floor;
    if (span <= 0) return 0;
    var norm = stats.lower
      ? (stats.floor - entry.mean) / span
      : (entry.mean - stats.floor) / span;
    return clamp01(norm);
  }

  function familyMembers(family) {
    return state.data.models.filter(function (m) {
      return m.category === family;
    });
  }

  /** Family members currently toggled on in the legend. */
  function enabledFamilyMembers(family) {
    return familyMembers(family).filter(function (m) {
      return state.enabledModels.has(m.name);
    });
  }

  /**
   * Average normalised radius across the *enabled* members of a family for a
   * task, so the aggregate line always reflects the current model selection.
   * Returns null when no enabled member has a value for the task.
   */
  function aggValue(task, family) {
    var members = enabledFamilyMembers(family);
    var sum = 0, count = 0;
    for (var i = 0; i < members.length; i++) {
      var v = normValue(task, members[i].name);
      if (v !== null) {
        sum += v;
        count++;
      }
    }
    return count ? sum / count : null;
  }

  // ------------------------------------------------------------------
  // Data-derived groupings
  // ------------------------------------------------------------------

  function tasksInGroup(group) {
    // Tasks whose category is in the group, ordered by the group's category
    // order and then by the task order within the payload.
    var order = {};
    group.categories.forEach(function (c, i) { order[c] = i; });
    return state.data.tasks
      .filter(function (t) { return order.hasOwnProperty(t.category); })
      .sort(function (a, b) { return order[a.category] - order[b.category]; });
  }

  function activeSeries() {
    // Ordered list of {kind, key, name, label, color, dashed}.
    var series = [];
    var models = state.data.models;
    for (var i = 0; i < models.length; i++) {
      var m = models[i];
      if (state.enabledModels.has(m.name)) {
        series.push({
          kind: "model",
          key: m.name,
          label: m.display,
          color: state.modelColor[m.name],
          dashed: false,
        });
      }
    }
    ["classic", "foundation"].forEach(function (fam) {
      if (state.aggEnabled[fam]) {
        series.push({
          kind: "agg",
          key: fam,
          label: AGG_LABELS[fam],
          color: AGG_COLORS[fam],
          dashed: true,
        });
      }
    });
    return series;
  }

  function seriesValue(series, task) {
    return series.kind === "agg"
      ? aggValue(task, series.key)
      : normValue(task, series.key);
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function render() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container || !state.data) return;

    var html = [];
    html.push('<div class="ncr-controls">');
    html.push(renderViewToggle());
    html.push(renderLegend());
    html.push("</div>");

    html.push('<div class="ncr-grid">');
    for (var c = 0; c < state.groups.length; c++) {
      html.push(renderCard(state.groups[c]));
    }
    html.push("</div>");

    container.innerHTML = html.join("");
    bindEvents(container);
  }

  function renderViewToggle() {
    var s = '<div class="ncr-view-toggle" role="group" aria-label="Metric">';
    s += '<button class="ncr-toggle-btn' + (state.view === "score" ? " ncr-active" : "") +
      '" data-view="score">Normalized score</button>';
    s += '<button class="ncr-toggle-btn' + (state.view === "rank" ? " ncr-active" : "") +
      '" data-view="rank">Rank</button>';
    s += "</div>";
    return s;
  }

  function renderLegend() {
    var s = '<div class="ncr-legend">';

    // Aggregate toggles first (headline feature).
    s += '<div class="ncr-legend-group ncr-legend-agg">';
    ["classic", "foundation"].forEach(function (fam) {
      var on = state.aggEnabled[fam];
      s += '<label class="ncr-chip ncr-chip-agg' + (on ? " ncr-on" : "") + '">';
      s += '<input type="checkbox" data-agg="' + fam + '"' + (on ? " checked" : "") + ">";
      s += '<span class="ncr-swatch ncr-swatch-dashed" style="--sw:' + AGG_COLORS[fam] + '"></span>';
      s += '<span class="ncr-chip-label">' + AGG_LABELS[fam] + " avg</span>";
      s += "</label>";
    });
    s += "</div>";

    // Per-family model chips.
    var families = [
      ["foundation", "Foundation"],
      ["classic", "Task-specific"],
      ["baseline", "Baselines"],
    ];
    families.forEach(function (fam) {
      var members = familyMembers(fam[0]);
      if (!members.length) return;
      s += '<div class="ncr-legend-group">';
      s += '<span class="ncr-group-title">' + fam[1] + "</span>";
      s += '<div class="ncr-group-chips">';
      for (var i = 0; i < members.length; i++) {
        var m = members[i];
        var on = state.enabledModels.has(m.name);
        s += '<label class="ncr-chip' + (on ? " ncr-on" : "") + '">';
        s += '<input type="checkbox" data-model="' + m.name + '"' + (on ? " checked" : "") + ">";
        s += '<span class="ncr-swatch" style="--sw:' + state.modelColor[m.name] + '"></span>';
        s += '<span class="ncr-chip-label">' + m.display + "</span>";
        s += "</label>";
      }
      s += "</div>";
      s += '<div class="ncr-group-actions">';
      s += '<button class="ncr-mini-btn" data-family-all="' + fam[0] + '">All</button>';
      s += '<button class="ncr-mini-btn" data-family-none="' + fam[0] + '">None</button>';
      s += "</div>";
      s += "</div>";
    });

    s += "</div>";
    return s;
  }

  function renderCard(group) {
    var tasks = tasksInGroup(group);
    if (!tasks.length) return "";

    var headline = foundationHeadline(tasks);
    var s = '<div class="ncr-card">';
    s += '<div class="ncr-card-head">';
    s += '<span class="ncr-card-title">' + escapeHtml(group.title) + "</span>";
    if (headline) {
      s += '<span class="ncr-card-metric" title="' +
        "Average across the enabled foundation models on this card's tasks" +
        '">' + headline + "</span>";
    }
    s += "</div>";
    s += renderRadar(tasks, group.title);
    s += "</div>";
    return s;
  }

  /**
   * Small headline per card: average over the *enabled* foundation models on
   * this card's tasks (mean rank in rank view, mean normalised score in score
   * view).  Empty when no enabled foundation model has a value here.
   */
  function foundationHeadline(tasks) {
    if (state.view === "rank") {
      var members = enabledFamilyMembers("foundation");
      var sum = 0, count = 0;
      for (var t = 0; t < tasks.length; t++) {
        for (var i = 0; i < members.length; i++) {
          var e = (state.data.results[tasks[t].id] || {})[members[i].name];
          if (e && typeof e.rank === "number") {
            sum += e.rank;
            count++;
          }
        }
      }
      return count ? "Foundation avg #" + (sum / count).toFixed(1) : "";
    }
    var s2 = 0, n2 = 0;
    for (var k = 0; k < tasks.length; k++) {
      var v = aggValue(tasks[k], "foundation");
      if (v !== null) {
        s2 += v;
        n2++;
      }
    }
    return n2 ? "Foundation avg " + (s2 / n2).toFixed(2) : "";
  }

  function renderRadar(tasks, groupTitle) {
    // Drawing centre is (150, 150); the viewBox is padded horizontally so
    // axis labels placed just outside the outer ring stay within the SVG box
    // (and therefore scale to fit rather than being clipped by the card edge).
    var CX = 150;
    var CY = 150;
    var R = 82;
    var VB = { x: -56, y: -16, w: 412, h: 336 };
    var colors = themeColors();
    var n = tasks.length;

    // Angle for axis i (start at top, clockwise).
    function angle(i) {
      return -Math.PI / 2 + (2 * Math.PI * i) / n;
    }
    function point(i, radius) {
      var a = angle(i);
      return [CX + radius * R * Math.cos(a), CY + radius * R * Math.sin(a)];
    }

    var viewLabel = state.view === "rank" ? "rank" : "normalized score";
    var aria = escapeHtml(
      (groupTitle ? groupTitle + " " : "") +
        "radar chart (" + viewLabel + "), one spoke per task"
    );
    var svg = [];
    svg.push('<svg class="ncr-svg" viewBox="' + VB.x + " " + VB.y + " " + VB.w + " " + VB.h +
      '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="' + aria + '">');

    // Grid rings.
    var rings = [0.25, 0.5, 0.75, 1.0];
    for (var r = 0; r < rings.length; r++) {
      var pts = [];
      for (var i = 0; i < n; i++) {
        var p = point(i, rings[r]);
        pts.push(p[0].toFixed(1) + "," + p[1].toFixed(1));
      }
      var shape = n >= 3
        ? '<polygon points="' + pts.join(" ") + '"'
        : '<polyline points="' + pts.join(" ") + '"';
      svg.push(shape + ' fill="none" stroke="' + colors.grid + '" stroke-width="1"/>');
    }

    // Axes + labels.
    for (var i = 0; i < n; i++) {
      var edge = point(i, 1.0);
      svg.push('<line x1="' + CX + '" y1="' + CY + '" x2="' + edge[0].toFixed(1) +
        '" y2="' + edge[1].toFixed(1) + '" stroke="' + colors.axis + '" stroke-width="1"/>');

      var lp = point(i, 1.18);
      var a = angle(i);
      var anchor = "middle";
      if (Math.cos(a) > 0.3) anchor = "start";
      else if (Math.cos(a) < -0.3) anchor = "end";
      var dy = Math.sin(a) > 0.3 ? "0.7em" : Math.sin(a) < -0.3 ? "-0.1em" : "0.3em";
      var label = escapeHtml(truncate(tasks[i].display, 13));
      svg.push('<text class="ncr-axis-label" x="' + lp[0].toFixed(1) + '" y="' +
        lp[1].toFixed(1) + '" text-anchor="' + anchor + '" dy="' + dy +
        '" fill="' + colors.label + '"><title>' + escapeHtml(tasks[i].display) +
        "</title>" + label + "</text>");
    }

    // Series polygons.
    var series = activeSeries();
    for (var s = 0; s < series.length; s++) {
      var seriesItem = series[s];
      var coords = [];
      var markers = [];
      for (var j = 0; j < n; j++) {
        var val = seriesValue(seriesItem, tasks[j]);
        var radius = val === null ? 0 : val;
        var pt = point(j, radius);
        coords.push(pt[0].toFixed(1) + "," + pt[1].toFixed(1));
        if (val !== null) {
          markers.push({ x: pt[0], y: pt[1], task: tasks[j] });
        }
      }
      var dash = seriesItem.dashed ? ' stroke-dasharray="5 3"' : "";
      var sw = seriesItem.kind === "agg" ? 2.6 : 1.6;
      var fill = seriesItem.kind === "agg" ? hexToRgba(seriesItem.color, 0.10) : "none";
      var shape2 = n >= 3
        ? '<polygon points="' + coords.join(" ") + '"'
        : '<polyline points="' + coords.join(" ") + '"';
      svg.push(shape2 + ' fill="' + fill + '" stroke="' + seriesItem.color +
        '" stroke-width="' + sw + '" stroke-linejoin="round"' + dash +
        ' opacity="' + (seriesItem.kind === "agg" ? 0.95 : 0.8) + '"/>');

      for (var mk = 0; mk < markers.length; mk++) {
        var m = markers[mk];
        var rad = seriesItem.kind === "agg" ? 3.2 : 2.4;
        svg.push('<circle class="ncr-marker" cx="' + m.x.toFixed(1) + '" cy="' +
          m.y.toFixed(1) + '" r="' + rad + '" fill="' + seriesItem.color +
          '" data-task="' + m.task.id + '" data-series="' + seriesItem.kind + ":" + seriesItem.key +
          '"></circle>');
      }
    }

    svg.push("</svg>");
    return svg.join("");
  }

  // ------------------------------------------------------------------
  // Tooltip
  // ------------------------------------------------------------------

  function ensureTooltip() {
    var tip = document.getElementById("ncr-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "ncr-tooltip";
      tip.className = "ncr-tooltip";
      tip.style.display = "none";
      document.body.appendChild(tip);
    }
    return tip;
  }

  function tooltipHtml(task, seriesId) {
    var parts = seriesId.split(":");
    var kind = parts[0];
    var key = parts.slice(1).join(":");
    var title, valueLine;
    if (kind === "agg") {
      title = AGG_LABELS[key] + " avg";
      var v = aggValue(task, key);
      valueLine = state.view === "rank"
        ? "Avg normalized rank: " + (v === null ? "—" : v.toFixed(2))
        : "Avg normalized score: " + (v === null ? "—" : v.toFixed(2));
    } else {
      var model = null;
      for (var i = 0; i < state.data.models.length; i++) {
        if (state.data.models[i].name === key) model = state.data.models[i];
      }
      title = model ? model.display : key;
      var entry = (state.data.results[task.id] || {})[key] || {};
      var digits =
        task.objective === "classification" || task.objective === "retrieval" ? 1 : 3;
      var raw = typeof entry.mean === "number" ? entry.mean.toFixed(digits) : "—";
      valueLine = task.metric_display + ": " + raw;
      if (typeof entry.std === "number" && entry.std > 0) {
        valueLine += " ± " + entry.std.toFixed(digits);
      }
      if (typeof entry.rank === "number") valueLine += "  ·  rank " + entry.rank;
    }
    return '<div class="ncr-tip-title">' + escapeHtml(title) + "</div>" +
      '<div class="ncr-tip-task">' + escapeHtml(task.display) + "</div>" +
      '<div class="ncr-tip-val">' + escapeHtml(valueLine) + "</div>";
  }

  // ------------------------------------------------------------------
  // Events
  // ------------------------------------------------------------------

  function bindEvents(container) {
    var viewBtns = container.querySelectorAll(".ncr-toggle-btn");
    for (var i = 0; i < viewBtns.length; i++) {
      viewBtns[i].addEventListener("click", function (e) {
        state.view = e.currentTarget.dataset.view;
        render();
      });
    }

    var modelBoxes = container.querySelectorAll('input[data-model]');
    for (var i = 0; i < modelBoxes.length; i++) {
      modelBoxes[i].addEventListener("change", function (e) {
        var name = e.currentTarget.dataset.model;
        if (e.currentTarget.checked) state.enabledModels.add(name);
        else state.enabledModels.delete(name);
        render();
      });
    }

    var aggBoxes = container.querySelectorAll('input[data-agg]');
    for (var i = 0; i < aggBoxes.length; i++) {
      aggBoxes[i].addEventListener("change", function (e) {
        state.aggEnabled[e.currentTarget.dataset.agg] = e.currentTarget.checked;
        render();
      });
    }

    var allBtns = container.querySelectorAll("[data-family-all]");
    for (var i = 0; i < allBtns.length; i++) {
      allBtns[i].addEventListener("click", function (e) {
        familyMembers(e.currentTarget.dataset.familyAll).forEach(function (m) {
          state.enabledModels.add(m.name);
        });
        render();
      });
    }
    var noneBtns = container.querySelectorAll("[data-family-none]");
    for (var i = 0; i < noneBtns.length; i++) {
      noneBtns[i].addEventListener("click", function (e) {
        familyMembers(e.currentTarget.dataset.familyNone).forEach(function (m) {
          state.enabledModels.delete(m.name);
        });
        render();
      });
    }

    bindMarkerTooltips(container);
  }

  function bindMarkerTooltips(container) {
    var tip = ensureTooltip();
    var markers = container.querySelectorAll(".ncr-marker");
    for (var i = 0; i < markers.length; i++) {
      markers[i].addEventListener("mouseenter", function (e) {
        var el = e.currentTarget;
        var task = taskById(el.dataset.task);
        if (!task) return;
        tip.innerHTML = tooltipHtml(task, el.dataset.series);
        tip.style.display = "block";
      });
      markers[i].addEventListener("mousemove", function (e) {
        tip.style.left = e.clientX + 14 + "px";
        tip.style.top = e.clientY + 14 + "px";
      });
      markers[i].addEventListener("mouseleave", function () {
        tip.style.display = "none";
      });
    }
  }

  function taskById(id) {
    for (var i = 0; i < state.data.tasks.length; i++) {
      if (state.data.tasks[i].id === id) return state.data.tasks[i];
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Small utilities
  // ------------------------------------------------------------------

  function truncate(s, max) {
    s = String(s);
    return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function hexToRgba(hex, alpha) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var r = parseInt(h.substring(0, 2), 16);
    var g = parseInt(h.substring(2, 4), 16);
    var b = parseInt(h.substring(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function observeDarkMode() {
    var observer = new MutationObserver(function () { render(); });
    observer.observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
    }
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------

  function resolveDataUrl() {
    var scripts = document.getElementsByTagName("script");
    for (var i = 0; i < scripts.length; i++) {
      var src = scripts[i].src || "";
      var idx = src.indexOf(SCRIPT_FILENAME);
      if (idx !== -1) return src.substring(0, idx) + DATA_FILENAME;
    }
    return "_static/" + DATA_FILENAME;
  }

  function initState(data) {
    state.data = data;
    state.groups = radarGroups(data);
    state.enabledModels = new Set();
    state.modelColor = {};
    for (var i = 0; i < data.models.length; i++) {
      var m = data.models[i];
      state.modelColor[m.name] = PALETTE[i % PALETTE.length];
      // Default: trainable models on, baselines and aggregates off.
      if (m.category === "classic" || m.category === "foundation") {
        state.enabledModels.add(m.name);
      }
    }
  }

  /**
   * Radar groups: one card per group.  Prefer the payload's ``radar_groups``
   * (computed in generate_results_json.py so small categories are merged into
   * radars with enough spokes); otherwise fall back to one card per category.
   */
  function radarGroups(data) {
    if (data.radar_groups && data.radar_groups.length) return data.radar_groups;
    return (data.categories || []).map(function (c) {
      return { title: c, categories: [c] };
    });
  }

  function init() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    fetch(resolveDataUrl())
      .then(function (r) { return r.json(); })
      .then(function (data) {
        initState(data);
        observeDarkMode();
        render();
      })
      .catch(function (err) {
        container.innerHTML =
          '<p style="color:#888;font-style:italic">Per-category radar plots could not be loaded.</p>';
        console.error("NeuralBench category radars: failed to load data", err);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
