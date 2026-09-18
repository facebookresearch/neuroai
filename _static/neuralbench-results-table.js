/**
 * Interactive benchmark results table for the NeuralBench landing page.
 *
 * Renders a heatmap-style pivot table (tasks x models) with filtering,
 * pagination, sorting, and a scores/ranks toggle.  Fetches data from
 * neuralbench-results-data.json which is generated either from mock data
 * or from real benchmark CSV outputs.
 */

(function () {
  "use strict";

  var CONTAINER_ID = "neuralbench-results-table";
  var SCRIPT_FILENAME = "neuralbench-results-table.js";
  var DATA_FILENAME = "neuralbench-results-data.json";

  var state = {
    data: null,
    view: "scores",         // "scores" | "ranks"
    device: "all",
    objective: "all",
    modelCategory: "all",
    rowsPerPage: "all",     // "all" | 10 | 20 | 50
    currentPage: 0,
    sortModel: null,        // model name key to sort by
    sortAsc: true,
    fullscreen: false,
  };

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  function isDarkMode() {
    var body = document.body;
    if (body.dataset.theme === "dark") return true;
    if (body.dataset.theme === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  /**
   * Map a normalised 0-1 value to a heatmap colour.
   * 0 = worst (red-ish), 1 = best (green-ish).
   * For regression (lower-is-better) the caller inverts before passing.
   */
  function heatmapColor(t, dark) {
    t = Math.max(0, Math.min(1, t));
    // HSL approach: hue goes from 0 (red) to 120 (green).
    var hue = Math.round(t * 120);
    if (dark) {
      return "hsla(" + hue + ", 55%, 28%, 0.65)";
    }
    return "hsla(" + hue + ", 60%, 85%, 0.75)";
  }

  function taskDocUrl(task) {
    return "tasks/" + task.device + "/" + task.name + ".html";
  }

  // ------------------------------------------------------------------
  // Filtering
  // ------------------------------------------------------------------

  function filteredTasks() {
    if (!state.data) return [];
    return state.data.tasks.filter(function (t) {
      if (state.device !== "all" && t.device !== state.device) return false;
      if (state.objective !== "all" && t.objective !== state.objective) return false;
      return true;
    });
  }

  function filteredModels() {
    if (!state.data) return [];
    return state.data.models.filter(function (m) {
      if (state.modelCategory !== "all" && m.category !== state.modelCategory) return false;
      return true;
    });
  }

  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------

  function sortTasks(tasks, models) {
    if (!state.sortModel) return tasks;
    var results = state.data.results;
    var key = state.view === "ranks" ? "rank" : "mean";
    var lowerBetter = state.view === "ranks";
    tasks.sort(function (a, b) {
      var ra = results[a.id] && results[a.id][state.sortModel];
      var rb = results[b.id] && results[b.id][state.sortModel];
      var va = ra ? ra[key] : (lowerBetter ? 9999 : -9999);
      var vb = rb ? rb[key] : (lowerBetter ? 9999 : -9999);
      var cmp = va - vb;
      // For regression scores, lower is better
      if (state.view === "scores" && a.objective === "regression") cmp = -cmp;
      return state.sortAsc ? cmp : -cmp;
    });
    return tasks;
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function render() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container || !state.data) return;

    var tasks = filteredTasks();
    var models = filteredModels();
    tasks = sortTasks(tasks.slice(), models);

    var totalTasks = tasks.length;
    var perPage = state.rowsPerPage === "all" ? totalTasks : state.rowsPerPage;
    var totalPages = Math.max(1, Math.ceil(totalTasks / perPage));
    if (state.currentPage >= totalPages) state.currentPage = totalPages - 1;
    var start = state.currentPage * perPage;
    var pageTasks = tasks.slice(start, start + perPage);

    var html = [];

    // Controls bar
    html.push('<div class="nrt-controls">');
    html.push(renderViewToggle());
    html.push(renderSelect("nrt-device", "Device", [
      ["all", "All devices"], ["eeg", "EEG"]
    ], state.device));
    html.push(renderSelect("nrt-objective", "Task type", [
      ["all", "All types"], ["classification", "Classification"],
      ["multilabel", "Multilabel"], ["regression", "Regression"],
      ["retrieval", "Retrieval"]
    ], state.objective));
    html.push(renderSelect("nrt-model-cat", "Models", [
      ["all", "All models"], ["classic", "Task-specific"], ["foundation", "Foundation"],
      ["baseline", "Baseline"]
    ], state.modelCategory));
    html.push(renderSelect("nrt-per-page", "Show", [
      ["all", "All rows"], ["10", "10 rows"], ["20", "20 rows"], ["50", "50 rows"]
    ], String(state.rowsPerPage)));
    html.push(renderFullscreenButton());
    html.push('</div>');

    // Table
    html.push('<div class="nrt-table-wrap">');
    html.push('<table class="nrt-table">');

    // Header
    html.push("<thead><tr>");
    html.push('<th class="nrt-th-task">Task</th>');
    html.push('<th class="nrt-th-device">Device</th>');
    html.push('<th class="nrt-th-metric">Metric</th>');
    for (var mi = 0; mi < models.length; mi++) {
      var m = models[mi];
      var sortCls = "nrt-th-model nrt-cat-" + m.category;
      if (state.sortModel === m.name) sortCls += " nrt-sorted";
      html.push('<th class="' + sortCls + '" data-model="' + m.name + '">');
      html.push('<span class="nrt-model-label">' + m.display + '</span>');
      if (state.sortModel === m.name) {
        html.push('<span class="nrt-sort-arrow">' + (state.sortAsc ? " \u25B2" : " \u25BC") + "</span>");
      }
      html.push("</th>");
    }
    html.push("</tr></thead>");

    // Body
    html.push("<tbody>");
    var dark = isDarkMode();
    for (var ti = 0; ti < pageTasks.length; ti++) {
      var task = pageTasks[ti];
      var row = state.data.results[task.id] || {};
      var isLowerBetter = task.lower_is_better === true;
      var isRank = state.view === "ranks";

      // Compute min/max for heatmap normalisation within this row
      var vals = [];
      for (var j = 0; j < models.length; j++) {
        var entry = row[models[j].name];
        if (entry) vals.push(isRank ? entry.rank : entry.mean);
      }
      var vmin = vals.length ? Math.min.apply(null, vals) : 0;
      var vmax = vals.length ? Math.max.apply(null, vals) : 1;
      var vrange = vmax - vmin || 1;

      // Find best value in row
      var bestVal = null;
      var bestModel = null;
      for (var j = 0; j < models.length; j++) {
        var entry = row[models[j].name];
        if (!entry) continue;
        var v = isRank ? entry.rank : entry.mean;
        if (bestVal === null ||
            (isRank && v < bestVal) ||
            (!isRank && !isLowerBetter && v > bestVal) ||
            (!isRank && isLowerBetter && v < bestVal)) {
          bestVal = v;
          bestModel = models[j].name;
        }
      }

      html.push("<tr>");
      html.push('<td class="nrt-td-task"><a href="' + taskDocUrl(task) + '">' + task.display + "</a></td>");
      html.push('<td class="nrt-td-device"><span class="nrt-badge nrt-badge-' + task.device + '">' + task.device.toUpperCase() + "</span></td>");
      html.push('<td class="nrt-td-metric">' + task.metric_display + "</td>");

      for (var j = 0; j < models.length; j++) {
        var entry = row[models[j].name];
        if (!entry) {
          html.push('<td class="nrt-td-value">—</td>');
          continue;
        }
        var val = isRank ? entry.rank : entry.mean;
        var norm = (val - vmin) / vrange;
        // Invert so that "best" = 1 (green):
        //   ranks & regression: lower is better -> invert
        //   scores (non-reg): higher is better -> keep
        if (isRank || isLowerBetter) norm = 1 - norm;

        var bg = heatmapColor(norm, dark);
        var isBest = models[j].name === bestModel;
        var cls = "nrt-td-value" + (isBest ? " nrt-best" : "");

        var displayVal;
        if (isRank) {
          displayVal = val;
        } else {
          displayVal = (task.objective === "multilabel" || task.objective === "regression")
            ? val.toFixed(3)
            : val.toFixed(1);
        }

        var tooltip = task.display + " — " + models[j].display + "\n";
        tooltip += task.metric_display + ": " + entry.mean;
        if (!isRank) tooltip += " \u00b1 " + entry.std;
        tooltip += "\nRank: " + entry.rank + " / " + models.length;

        html.push('<td class="' + cls + '" style="background:' + bg + '" title="' + tooltip + '">' + displayVal + "</td>");
      }
      html.push("</tr>");
    }

    // Average rank row (only in ranks view)
    if (state.view === "ranks" && pageTasks.length > 0) {
      html.push('<tr class="nrt-avg-row">');
      html.push('<td class="nrt-td-task nrt-avg-label" colspan="3">Average rank (visible tasks)</td>');
      for (var j = 0; j < models.length; j++) {
        var sum = 0, count = 0;
        for (var ti2 = 0; ti2 < tasks.length; ti2++) {
          var r = state.data.results[tasks[ti2].id];
          if (r && r[models[j].name]) {
            sum += r[models[j].name].rank;
            count++;
          }
        }
        var avg = count ? (sum / count) : 0;
        html.push('<td class="nrt-td-value nrt-avg-val">' + avg.toFixed(1) + "</td>");
      }
      html.push("</tr>");
    }

    html.push("</tbody></table></div>");

    // Pagination
    if (totalPages > 1) {
      html.push('<div class="nrt-pagination">');
      html.push('<button class="nrt-page-btn" data-page="prev"' + (state.currentPage === 0 ? " disabled" : "") + '>\u25C0 Prev</button>');
      html.push('<span class="nrt-page-info">Page ' + (state.currentPage + 1) + " / " + totalPages + " (" + totalTasks + " tasks)</span>");
      html.push('<button class="nrt-page-btn" data-page="next"' + (state.currentPage >= totalPages - 1 ? " disabled" : "") + '>Next \u25B6</button>');
      html.push("</div>");
    } else {
      html.push('<div class="nrt-pagination"><span class="nrt-page-info">' + totalTasks + " tasks</span></div>");
    }

    container.innerHTML = html.join("");
    bindEvents(container);
  }

  function renderViewToggle() {
    var s = '<div class="nrt-view-toggle">';
    s += '<button class="nrt-toggle-btn' + (state.view === "scores" ? " nrt-active" : "") + '" data-view="scores">Scores</button>';
    s += '<button class="nrt-toggle-btn' + (state.view === "ranks" ? " nrt-active" : "") + '" data-view="ranks">Ranks</button>';
    s += "</div>";
    return s;
  }

  function renderFullscreenButton() {
    // U+26F6 SQUARE FOUR CORNERS (enter), U+2715 MULTIPLICATION X (exit).
    var label = state.fullscreen ? "\u2715 Exit full view" : "\u26F6 Full view";
    var title = state.fullscreen ? "Exit full view (Esc)" : "Expand to full view";
    var cls = "nrt-fullscreen-btn" + (state.fullscreen ? " nrt-active" : "");
    return '<button class="' + cls + '" data-fullscreen="toggle" title="' + title + '">' + label + "</button>";
  }

  function renderSelect(id, label, options, current) {
    var s = '<label class="nrt-filter-label">' + label + " ";
    s += '<select class="nrt-select" id="' + id + '">';
    for (var i = 0; i < options.length; i++) {
      var sel = options[i][0] === current ? " selected" : "";
      s += '<option value="' + options[i][0] + '"' + sel + ">" + options[i][1] + "</option>";
    }
    s += "</select></label>";
    return s;
  }

  // ------------------------------------------------------------------
  // Event binding
  // ------------------------------------------------------------------

  function bindEvents(container) {
    // View toggle
    var toggleBtns = container.querySelectorAll(".nrt-toggle-btn");
    for (var i = 0; i < toggleBtns.length; i++) {
      toggleBtns[i].addEventListener("click", function (e) {
        state.view = e.target.dataset.view;
        state.currentPage = 0;
        render();
      });
    }

    // Select filters
    bindSelect("nrt-device", "device");
    bindSelect("nrt-objective", "objective");
    bindSelect("nrt-model-cat", "modelCategory");

    var perPage = container.querySelector("#nrt-per-page");
    if (perPage) {
      perPage.addEventListener("change", function () {
        state.rowsPerPage = this.value === "all" ? "all" : parseInt(this.value, 10);
        state.currentPage = 0;
        render();
      });
    }

    // Column sorting
    var modelHeaders = container.querySelectorAll(".nrt-th-model");
    for (var i = 0; i < modelHeaders.length; i++) {
      modelHeaders[i].addEventListener("click", function () {
        var model = this.dataset.model;
        if (state.sortModel === model) {
          state.sortAsc = !state.sortAsc;
        } else {
          state.sortModel = model;
          state.sortAsc = true;
        }
        render();
      });
    }

    // Pagination
    var pageBtns = container.querySelectorAll(".nrt-page-btn");
    for (var i = 0; i < pageBtns.length; i++) {
      pageBtns[i].addEventListener("click", function () {
        if (this.dataset.page === "prev" && state.currentPage > 0) state.currentPage--;
        if (this.dataset.page === "next") state.currentPage++;
        render();
      });
    }

    // Fullscreen toggle
    var fsBtn = container.querySelector(".nrt-fullscreen-btn");
    if (fsBtn) {
      fsBtn.addEventListener("click", function () {
        setFullscreen(!state.fullscreen);
      });
    }
  }

  function setFullscreen(on) {
    state.fullscreen = !!on;
    var container = document.getElementById(CONTAINER_ID);
    if (container) container.classList.toggle("nrt-fullscreen", state.fullscreen);
    document.body.classList.toggle("nrt-fullscreen-lock", state.fullscreen);
    render();
  }

  function bindSelect(id, stateKey) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", function () {
      state[stateKey] = this.value;
      state.currentPage = 0;
      render();
    });
  }

  // ------------------------------------------------------------------
  // Dark mode observer
  // ------------------------------------------------------------------

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
    // Derive the JSON URL from this script's own <script src=...> so it
    // works regardless of which docs subfolder the page lives in
    // (e.g. /_internal_neuralbench/, /neuralbench/, or the site root).
    var scripts = document.getElementsByTagName("script");
    for (var i = 0; i < scripts.length; i++) {
      var src = scripts[i].src || "";
      var idx = src.indexOf(SCRIPT_FILENAME);
      if (idx !== -1) {
        return src.substring(0, idx) + DATA_FILENAME;
      }
    }
    return "_static/" + DATA_FILENAME;
  }

  function init() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    var dataUrl = resolveDataUrl();

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && state.fullscreen) {
        setFullscreen(false);
      }
    });

    fetch(dataUrl)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        state.data = data;
        observeDarkMode();
        render();
      })
      .catch(function (err) {
        container.innerHTML = '<p style="color:#888;font-style:italic">Benchmark results table could not be loaded.</p>';
        console.error("NeuralBench results table: failed to load data", err);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
