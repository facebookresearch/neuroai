// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

const initializedRoots = new WeakSet();

export function highlightSnippet(element, source, language) {
  const pattern = language === "python"
    ? /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:import|from|as|print|dict|True|False|None)\b/g
    : /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:pip|install|python)\b/g;
  const nodes = [];
  let offset = 0;
  for (const match of source.matchAll(pattern)) {
    nodes.push(element.ownerDocument.createTextNode(source.slice(offset, match.index)));
    const token = element.ownerDocument.createElement("span");
    const kind = match[0].startsWith("#") ? "comment"
      : /^["']/.test(match[0]) ? "string" : "keyword";
    token.className = `token ${kind}`;
    token.textContent = match[0];
    nodes.push(token);
    offset = match.index + match[0].length;
  }
  nodes.push(element.ownerDocument.createTextNode(source.slice(offset)));
  element.replaceChildren(...nodes);
}

export function initializeStudyExplorer(root = globalThis.document?.getElementById("study-explorer")) {
  if (!root || initializedRoots.has(root)) {
    return;
  }
  const chips = Array.from(root.querySelectorAll(".event-chip"));
  const tabs = Array.from(root.querySelectorAll(".explorer-tab"));
  const panels = Array.from(root.querySelectorAll(".explorer-tab-panel"));
  const eventHeaders = Array.from(root.querySelectorAll("th.event-col"));
  const studyCells = new Map(Array.from(root.querySelectorAll(".study-name"),
    (cell) => [cell.dataset.name, cell]));
  const tooltip = root.querySelector("#study-tooltip");
  const details = root.querySelector("#study-details");
  const viewNav = root.querySelector(".study-view-nav");
  const viewBack = root.querySelector(".study-view-back");
  const viewEmpty = root.querySelector(".study-view-empty");
  if (!details || !tooltip) {
    return;
  }
  initializedRoots.add(root);
  const detailsName = details.querySelector(".study-details-name");
  const detailsDeviceBadge = details.querySelector(".study-details-device-badge");
  const detailsAliases = details.querySelector(".study-details-aliases");
  const detailsDescription = details.querySelector(".study-details-description");
  const detailsStats = details.querySelector(".study-details-stats");
  const detailsEvents = details.querySelector(".study-details-events");
  const detailsEventsList = details.querySelector(".study-details-events-list");
  const detailsBashCode = details.querySelector(
    ".study-details-snippet--bash code"
  );
  const detailsPyCode = details.querySelector(
    ".study-details-snippet--python code"
  );
  const detailsOpen = details.querySelector(".study-details-open");
  const detailsGithub = details.querySelector(".study-details-github");
  const detailsReportsLink = details.querySelector(".study-details-reports-link");
  // Within each filter set we OR; across packages, devices and events we AND.
  const ALL_PACKAGES = chips
    .filter((c) => c.dataset.category === "package")
    .map((c) => c.dataset.event);
  const activePackages = new Set(ALL_PACKAGES);
  const PACKAGES_SET = new Set(ALL_PACKAGES);
  const ALL_DEVICES = chips
    .filter((c) => c.dataset.category === "device")
    .map((c) => c.dataset.event);
  const ALL_EVENTS = chips
    .filter((c) => c.dataset.category === "event")
    .map((c) => c.dataset.event);
  const activeDevices = new Set(ALL_DEVICES);
  const activeEvents = new Set(ALL_EVENTS);
  // Click vs. double-click disambiguation — a single click toggles, a
  // double click solos (or restores all if already soloed). Native dblclick
  // fires AFTER click, so we defer the click action briefly and cancel it
  // when a double-click follows.
  const CLICK_DELAY_MS = 220;
  let pendingClickTimer = null;
  let tooltipHideTimer = null;
  // Tracks the currently displayed study so a second click on the same
  // element collapses the details panel.
  let currentSelection = "";
  let detailsTrigger = null;
  let studyView = false;

  const NEURO_COLORS = JSON.parse(root.dataset.deviceColors);
  const INSTALL_COMMANDS = JSON.parse(root.dataset.installCommands);
  // Chip categoriser: the device row also exposes a synthetic "Multimodal"
  // chip. Studies that declare more than one neuro modality carry an extra
  // "Multimodal" entry in their `data-events`, so the same filter logic
  // matches them via this set membership.
  const NEURO_TYPES_SET = new Set([
    "Eeg", "Meg", "Fmri", "Ieeg", "Emg", "Fnirs", "Multimodal",
  ]);
  function elementEvents(element) {
    return (element.dataset.events || "").split("|").filter(Boolean);
  }

  function matchesFilters(element) {
    const events = elementEvents(element);
    const devicePass = activeDevices.size === 0 || activeDevices.size === ALL_DEVICES.length
      || events.some((e) => activeDevices.has(e));
    const eventPass = activeEvents.size === 0 || activeEvents.size === ALL_EVENTS.length
      || events.some((e) => activeEvents.has(e));
    const packagePass = activePackages.size === 0 || activePackages.size === ALL_PACKAGES.length
      || activePackages.has(element.dataset.package || "");
    return packagePass && devicePass && eventPass;
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) {
      return "";
    }
    if (value >= 1_000_000) {
      return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    if (value >= 100) {
      return value.toFixed(0);
    }
    if (value >= 10) {
      return value.toFixed(1);
    }
    return value.toFixed(2);
  }

  function updateSummary() {
    let studies = 0;
    let subjects = 0;
    let timelines = 0;
    let hours = 0;
    root.querySelectorAll(".study-row").forEach((row) => {
      if (row.hasAttribute("hidden")) {
        return;
      }
      studies += 1;
      subjects += Number(row.dataset.subjects || 0);
      timelines += Number(row.dataset.timelines || 0);
      const rowHours = Number(row.dataset.hours || 0);
      if (Number.isFinite(rowHours)) {
        hours += rowHours;
      }
    });
    root.querySelector("#summary-studies").textContent = studies.toLocaleString();
    root.querySelector("#summary-subjects").textContent = subjects.toLocaleString();
    root.querySelector("#summary-timelines").textContent = timelines.toLocaleString();
    root.querySelector("#summary-hours").textContent = formatNumber(hours);
  }

  function setForName(name) {
    if (PACKAGES_SET.has(name)) {
      return activePackages;
    }
    return NEURO_TYPES_SET.has(name) ? activeDevices : activeEvents;
  }

  function reflectFilter(element, name, button = element) {
    const set = setForName(name);
    const pressed = set.size === 0 || set.has(name);
    element.classList.toggle("active", pressed);
    button?.setAttribute("aria-pressed", String(pressed));
  }

  function applyFilters() {
    root.querySelectorAll(".study-row, .study-point").forEach((element) => {
      if (matchesFilters(element)) {
        element.removeAttribute("hidden");
      } else {
        element.setAttribute("hidden", "");
      }
    });
    eventHeaders.forEach((header) => {
      if (header.dataset.event) reflectFilter(header, header.dataset.event, header.querySelector("button"));
    });
    chips.forEach((chip) => {
      if (chip.dataset.event) reflectFilter(chip, chip.dataset.event);
    });
    updateSummary();
  }

  function allForName(name) {
    if (PACKAGES_SET.has(name)) {
      return ALL_PACKAGES;
    }
    return NEURO_TYPES_SET.has(name) ? ALL_DEVICES : ALL_EVENTS;
  }

  function toggleEvent(eventName) {
    if (!eventName) {
      return;
    }
    const set = setForName(eventName);
    if (!set.size) allForName(eventName).forEach((name) => set.add(name));
    if (set.has(eventName)) {
      set.delete(eventName);
    } else {
      set.add(eventName);
    }
    applyFilters();
  }

  function soloEvent(eventName) {
    if (!eventName) {
      return;
    }
    const set = setForName(eventName);
    const all = allForName(eventName);
    // Second double-click on the soloed chip restores the full selection
    // — matches Plotly-style legend behavior so users can get back to "all"
    // without hunting for every chip individually.
    if (set.size === 1 && set.has(eventName)) {
      set.clear();
      all.forEach((v) => set.add(v));
    } else {
      set.clear();
      set.add(eventName);
    }
    applyFilters();
  }

  function scheduleClick(eventName) {
    if (pendingClickTimer) {
      clearTimeout(pendingClickTimer);
    }
    pendingClickTimer = setTimeout(() => {
      pendingClickTimer = null;
      toggleEvent(eventName);
    }, CLICK_DELAY_MS);
  }

  function cancelPendingClick() {
    if (pendingClickTimer) {
      clearTimeout(pendingClickTimer);
      pendingClickTimer = null;
    }
  }

  function fillStats(target) {
    detailsStats.replaceChildren();
    const subjects = Number(target.dataset.subjects || 0);
    const timelines = Number(target.dataset.timelines || 0);
    const hours = Number(target.dataset.hours || 0);
    const frequency = Number(target.dataset.frequency || 0);
    const channels = Number(target.dataset.channels || 0);
    const neuro = target.dataset.neuro || target.dataset.device || "";
    const stats = [];
    if (target.dataset.package) {
      stats.push(["Package", target.dataset.package]);
    }
    if (neuro) {
      stats.push(["Modality", neuro]);
    }
    if (subjects > 0) {
      stats.push(["Subjects", subjects.toLocaleString()]);
    }
    if (timelines > 0) {
      stats.push(["Timelines", timelines.toLocaleString()]);
    }
    if (Number.isFinite(hours) && hours > 0) {
      stats.push(["Est. hours", formatNumber(hours)]);
    }
    if (channels > 0) {
      stats.push(["Channels", channels.toLocaleString()]);
    }
    if (frequency > 0) {
      stats.push(["Frequency", `${formatNumber(frequency)} Hz`]);
    }
    stats.forEach(([label, value]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      row.appendChild(dt);
      row.appendChild(dd);
      detailsStats.appendChild(row);
    });
    detailsStats.hidden = stats.length === 0;
  }

  function fillEvents(target) {
    // "Multimodal" is a derived device category we inject into
    // `data-events` for chip-filtering only — hide it from the actual
    // event-type pill list so users don't see it next to real events.
    const events = elementEvents(target).filter((e) => e !== "Multimodal");
    detailsEventsList.replaceChildren();
    if (events.length === 0) {
      detailsEvents.setAttribute("hidden", "");
      return;
    }
    events.forEach((eventName) => {
      const pill = document.createElement("span");
      pill.className = "event-pill";
      pill.textContent = eventName;
      detailsEventsList.appendChild(pill);
    });
    detailsEvents.removeAttribute("hidden");
  }

  function buildBashSnippet(requirements, pkg) {
    const lines = [INSTALL_COMMANDS[pkg] || `pip install ${pkg}`];
    const reqs = (requirements || "")
      .split("|")
      .map((r) => r.trim())
      .filter(Boolean);
    if (reqs.length) {
      const quoted = reqs.map((requirement) => "'" + requirement.replaceAll("'", "'\"'\"'") + "'");
      lines.push(`pip install ${quoted.join(" ")}`);
    }
    return lines.join("\n");
  }


  function buildPythonSnippet(name) {
    return [
      "import neuralset as ns",
      "",
      "study = ns.Study(",
      `    name=${JSON.stringify(name)},`,
      '    path="/data",  # where original study is downloaded',
      '    infra=dict(folder="/cache"),  # cache study.run() output',
      ")",
      "print(study.requirements)",
      "study.download()",
      "events = study.run()",
    ].join("\n");
  }

  function announceStudy() {
    root.dataset.selectedStudy = currentSelection;
    root.querySelectorAll("[data-action='study']").forEach((element) => {
      const selected = element.dataset.name === currentSelection;
      element.classList.toggle("is-selected", selected);
      const button = element.matches("g")
        ? element : element.querySelector(".study-link");
      if (button) button.setAttribute("aria-pressed", String(selected));
    });
    root.dispatchEvent(new CustomEvent("neuralreport-study-selected", {
      detail: { study: currentSelection || null }, bubbles: true,
    }));
  }

  function showDetails(target, trigger = null) {
    target = studyCells.get(target.dataset.name);
    const name = target.dataset.name || "";
    // Clicking the same study a second time closes the panel — mirrors the
    // dismiss-by-second-click pattern that maps studies and tables already
    // use elsewhere in the docs.
    if (!studyView && currentSelection === name && !details.hasAttribute("hidden")) {
      hideDetails(true);
      return;
    }
    detailsTrigger = trigger;
    currentSelection = name;
    const aliases = target.dataset.aliases || "";
    const description = target.dataset.description || "No description available.";
    const url = target.dataset.url || "";
    const device = target.dataset.device || "";
    detailsName.textContent = name || "Study";
    if (device && NEURO_COLORS[device]) {
      detailsDeviceBadge.textContent = device;
      detailsDeviceBadge.style.background = NEURO_COLORS[device];
      detailsDeviceBadge.removeAttribute("hidden");
    } else {
      detailsDeviceBadge.setAttribute("hidden", "");
      detailsDeviceBadge.style.background = "";
    }
    if (aliases) {
      detailsAliases.textContent = `a.k.a. ${aliases}`;
      detailsAliases.removeAttribute("hidden");
    } else {
      detailsAliases.setAttribute("hidden", "");
    }
    detailsDescription.textContent = description;
    fillStats(target);
    fillEvents(target);
    const bashSrc = buildBashSnippet(
      target.dataset.requirements || "", target.dataset.package || ""
    );
    const pySrc = buildPythonSnippet(name);
    highlightSnippet(detailsBashCode, bashSrc, "bash");
    highlightSnippet(detailsPyCode, pySrc, "python");
    if (url) {
      detailsOpen.href = url;
      detailsOpen.removeAttribute("hidden");
    } else {
      detailsOpen.setAttribute("hidden", "");
      detailsOpen.removeAttribute("href");
    }
    const githubUrl = target.dataset.codeUrl || "";
    if (githubUrl) {
      detailsGithub.href = githubUrl;
      detailsGithub.removeAttribute("hidden");
    } else {
      detailsGithub.setAttribute("hidden", "");
      detailsGithub.removeAttribute("href");
    }
    if (detailsReportsLink) {
      const reportUrl = new URL(root.dataset.reportIndexUrl, document.baseURI);
      reportUrl.searchParams.set("study", name);
      detailsReportsLink.href = reportUrl.href;
      detailsReportsLink.removeAttribute("hidden");
    }
    details.removeAttribute("hidden");
    announceStudy();
    hideTooltip(true);
    if (!studyView) {
      requestAnimationFrame(() => {
        details.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
  }

  function syncStudyView() {
    const url = new URL(window.location.href);
    const requested = url.searchParams.get("study");
    const wasStudyView = studyView;
    studyView = requested !== null;
    root.toggleAttribute("data-study-view", studyView);
    viewNav.hidden = !studyView;
    viewEmpty.hidden = !studyView || studyCells.has(requested);
    if (studyView) {
      url.searchParams.delete("study");
      viewBack.href = url.href;
      if (studyCells.has(requested)) {
        if (currentSelection !== requested || details.hidden) {
          showDetails(studyCells.get(requested));
        }
      } else {
        hideDetails();
      }
      hideTooltip(true);
    } else if (wasStudyView) {
      hideDetails();
    }
  }

  function hideDetails(restoreFocus = false) {
    const trigger = restoreFocus && details.contains(document.activeElement) ? detailsTrigger : null;
    details.setAttribute("hidden", "");
    detailsTrigger = null;
    currentSelection = "";
    announceStudy();
    if (trigger?.isConnected && trigger.getClientRects().length) {
      trigger.focus();
    } else if (trigger) {
      tabs.find((tab) => tab.getAttribute("aria-selected") === "true")?.focus();
    }
  }

  function moveTooltip(event) {
    const margin = 14;
    const rect = tooltip.getBoundingClientRect();
    let left = event.clientX + margin;
    let top = event.clientY + margin;
    if (left + rect.width > window.innerWidth) {
      left = event.clientX - rect.width - margin;
    }
    if (top + rect.height > window.innerHeight) {
      top = event.clientY - rect.height - margin;
    }
    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function showTooltip(event) {
    const target = studyCells.get(event.currentTarget.dataset.name);
    const name = target.dataset.name || "";
    const aliases = target.dataset.aliases || "";
    const description = target.dataset.description || "No description available.";
    clearTimeout(tooltipHideTimer);
    tooltip.replaceChildren();
    if (name !== "") {
      const title = document.createElement("div");
      title.className = "tooltip-title";
      title.textContent = name;
      tooltip.appendChild(title);
    }
    const text = document.createElement("div");
    text.textContent = description;
    tooltip.appendChild(text);
    if (aliases !== "") {
      const aliasText = document.createElement("div");
      aliasText.textContent = `Aliases: ${aliases}`;
      tooltip.appendChild(aliasText);
    }
    const hint = document.createElement("div");
    hint.className = "study-tooltip-hint";
    hint.textContent = "Click for full details";
    tooltip.appendChild(hint);
    tooltip.removeAttribute("hidden");
    moveTooltip(event);
  }

  function hideTooltip(immediate) {
    clearTimeout(tooltipHideTimer);
    if (immediate) {
      tooltip.setAttribute("hidden", "");
      return;
    }
    tooltipHideTimer = setTimeout(() => {
      tooltip.setAttribute("hidden", "");
    }, 120);
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => scheduleClick(chip.dataset.event || ""));
    chip.addEventListener("dblclick", () => {
      cancelPendingClick();
      soloEvent(chip.dataset.event || "");
    });
  });
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab || ""));
    tab.addEventListener("keydown", (event) => {
      const key = event.key;
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(key)) return;
      event.preventDefault();
      const idx = tabs.indexOf(tab);
      const next = key === "Home" ? tabs[0] : key === "End" ? tabs.at(-1)
        : tabs[(idx + (key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
      activateTab(next.dataset.tab || "");
      next.focus();
    });
  });

  function activateTab(name) {
    if (!name) {
      return;
    }
    tabs.forEach((tab) => {
      const isActive = tab.dataset.tab === name;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
      tab.setAttribute("tabindex", isActive ? "0" : "-1");
    });
    panels.forEach((panel) => {
      if (panel.dataset.panel === name) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
    });
  }

  eventHeaders.forEach((header) => {
    header.addEventListener("click", () => scheduleClick(header.dataset.event || ""));
    header.addEventListener("dblclick", () => {
      cancelPendingClick();
      soloEvent(header.dataset.event || "");
    });
  });
  root.querySelectorAll("[data-action='study']").forEach((element) => {
    element.addEventListener("click", (event) => {
      // Allow event-col header clicks (which propagate through their <button>)
      // to keep their toggle behavior; only study cells / circles open the panel.
      if (event.target.closest("th.event-col")) {
        return;
      }
      showDetails(element, element.matches("g") ? element : element.querySelector(".study-link"));
    });
    if (element.matches("g")) {
      element.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          if (!event.repeat) showDetails(element, element);
        }
      });
    }
    element.addEventListener("mouseenter", showTooltip);
    element.addEventListener("mousemove", moveTooltip);
    element.addEventListener("mouseleave", () => hideTooltip(false));
  });
  details.querySelectorAll("[data-details-close]").forEach((el) => {
    el.addEventListener("click", () => {
      if (!studyView) hideDetails(true);
    });
  });
  document.addEventListener("keydown", (event) => {
    if (!studyView && event.key === "Escape" && details.getClientRects().length
        && !document.querySelector("dialog[open]")) {
      hideDetails(true);
    }
  });
  document.addEventListener("alpine:initialized", announceStudy);
  window.addEventListener("popstate", syncStudyView);
  applyFilters();
  syncStudyView();
  announceStudy();
}

initializeStudyExplorer();
