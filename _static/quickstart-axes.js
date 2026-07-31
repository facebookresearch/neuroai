/* quickstart-axes.js — Inline mini-version of the Code Builder axis bar
   used on the NeuralSet landing page (docs/neuralset/index.rst).

   Visually identical to the dropdowns on the full Code Builder page, but
   without the rendered code blocks — picking values here just updates the
   trailing "Open in Code Builder" link, which navigates to
   `code_builder.html?<axis>=<key>&...`. The destination page picks up the
   query params via the matching `URLSearchParams` block in code-builder.js
   and lands on a fully configured pipeline.

   Reads `window.CB_DATA` (same YAML source as code-builder.js). Each
   `.qs-builder` root carries:
     - `data-axes`     — comma-separated axis ids to expose (default:
                         `neuro,stim,task,compute`).
     - `data-defaults` — comma-separated `axis=key` overrides on top of the
                         per-axis YAML defaults.
   Unknown axes / option keys are ignored silently so the page survives
   stale defaults if the YAML evolves. */

(function () {
  if (typeof window === "undefined") return;

  function init() {
    var roots = document.querySelectorAll(".qs-builder");
    if (!roots.length || !window.CB_DATA) return;
    roots.forEach(buildOne);
  }

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function parseKVList(s) {
    var out = {};
    (s || "").split(",").forEach(function (kv) {
      kv = kv.trim();
      if (!kv) return;
      var eq = kv.indexOf("=");
      if (eq < 0) return;
      out[kv.slice(0, eq).trim()] = kv.slice(eq + 1).trim();
    });
    return out;
  }

  function buildOne(root) {
    var DATA = window.CB_DATA;
    var AXES = DATA.axes;

    var axes = (root.dataset.axes || "neuro,stim,task,model,compute")
      .split(",").map(function (s) { return s.trim(); }).filter(Boolean)
      .filter(function (a) { return !!AXES[a]; });

    var defaults = parseKVList(root.dataset.defaults || "");
    var sel = {};
    axes.forEach(function (a) {
      var dv = defaults[a];
      sel[a] = (dv && AXES[a].options[dv]) ? dv : AXES[a].default;
    });

    // ── Bar of axis dropdowns ────────────────────────────────────────────
    var bar = el("div", "qs-builder-bar");
    var axisWraps = {};

    axes.forEach(function (axis) {
      var meta = AXES[axis];
      var wrap = el("div", "cb-axis");
      wrap.dataset.axis = axis;
      axisWraps[axis] = wrap;

      var summary = el("button", "cb-axis-summary");
      summary.type = "button";
      summary.setAttribute("aria-expanded", "false");
      summary.innerHTML =
        '<span class="cb-axis-label">' + escapeHtml(meta.label) + '</span>';
      var sval = el("span", "cb-axis-summary-value");
      summary.appendChild(sval);
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
          sval.textContent = opt.label;
          syncHref();
          syncModelVisibility();
          closeTray();
        });
        pillByKey[key] = pill;
        tray.appendChild(pill);
      });
      wrap.appendChild(tray);

      sval.textContent = meta.options[sel[axis]].label;

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
          // One-at-a-time disclosure inside this builder.
          bar.querySelectorAll(".cb-axis.cb-open").forEach(function (a) {
            a.classList.remove("cb-open");
            var s = a.querySelector(".cb-axis-summary");
            if (s) s.setAttribute("aria-expanded", "false");
          });
          openTray();
        }
      });

      bar.appendChild(wrap);
    });

    // ── Trailing "Open in Code Builder" link ─────────────────────────────
    // The href is rebuilt on every pill click so the link always reflects
    // the current selection. The destination page parses the query string
    // and preselects the matching options.
    var goBtn = el(
      "a", "qs-builder-go",
      'Show me! <span aria-hidden="true">&rarr;</span>'
    );
    function syncHref() {
      var qs = axes.map(function (a) {
        return a + "=" + encodeURIComponent(sel[a]);
      }).join("&");
      goBtn.href = "code_builder.html" + (qs ? "?" + qs : "");
    }
    // Mirrors the matching guard in code-builder.js: the Training axis
    // (`model`) only matters when a task actually trains something. Hide
    // it whenever Task is "Load only" so the bar stays uncluttered, and
    // bring it back as soon as Decoding/Encoding is picked.
    function syncModelVisibility() {
      if (!axisWraps.model || !axisWraps.task) return;
      var taskOpt = AXES.task.options[sel.task] || {};
      axisWraps.model.style.display = taskOpt.needs_ml ? "" : "none";
    }
    syncHref();
    syncModelVisibility();

    root.appendChild(bar);
    root.appendChild(goBtn);

    // Click outside any axis → close every open tray inside this builder.
    document.addEventListener("click", function (ev) {
      if (!root.contains(ev.target)) {
        bar.querySelectorAll(".cb-axis.cb-open").forEach(function (a) {
          a.classList.remove("cb-open");
          var s = a.querySelector(".cb-axis-summary");
          if (s) s.setAttribute("aria-expanded", "false");
        });
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
