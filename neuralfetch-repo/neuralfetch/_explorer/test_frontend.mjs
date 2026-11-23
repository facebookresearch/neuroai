// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import assert from "node:assert/strict";
import { test } from "node:test";
import { initializeStudyExplorer, highlightSnippet } from "./static/study_explorer.js";

test("module initialization is idempotent and unknown queries stay out of the catalog", (t) => {
  assert.equal(initializeStudyExplorer(null), undefined);
  assert.equal(initializeStudyExplorer(), undefined);
  const originalDocument = globalThis.document;
  const originalWindow = globalThis.window;
  let queries = 0;
  let handlers = 0;
  let popstate;
  const node = () => ({ style: {}, textContent: "", setAttribute() {}, querySelector: node, querySelectorAll: () => [] });
  const view = new Map(["nav", "back", "empty"].map((name) => [`.study-view-${name}`, node()]));
  const attrs = {};
  const chipHandlers = {};
  let active = false;
  const chip = {
    dataset: { event: "Eeg", category: "device" },
    classList: { toggle(name, value) { assert.equal(name, "active"); active = value; } },
    setAttribute(name, value) { attrs[name] = value; },
    addEventListener(name, fn) { chipHandlers[name] = fn; },
  };
  const rows = ["Eeg|Word", "Eeg"].map((events) => ({
    dataset: { events }, hidden: false,
    setAttribute(name) { if (name === "hidden") this.hidden = true; },
    removeAttribute(name) { if (name === "hidden") this.hidden = false; },
    hasAttribute(name) { return name === "hidden" && this.hidden; },
  }));
  const wordAttrs = {}, wordHandlers = {};
  const word = {
    dataset: { event: "Word", category: "event" },
    classList: { toggle() {} },
    setAttribute(name, value) { wordAttrs[name] = value; },
    addEventListener(name, fn) { wordHandlers[name] = fn; },
  };
  const root = {
    dataset: { deviceColors: "{}", installCommands: "{}" },
    querySelectorAll(selector) {
      queries++;
      if (selector === ".event-chip") return [chip, word];
      return selector === ".study-row, .study-point" || selector === ".study-row" ? rows : [];
    },
    querySelector(selector) { queries++; return view.get(selector) || node(); },
    toggleAttribute(name, value) { this[name] = value; },
    dispatchEvent() {},
  };
  const url = new URL("https://example.org/explore.html?tab=table#catalog");
  globalThis.document = { addEventListener() { handlers++; } };
  globalThis.window = { location: url, addEventListener(name, fn) {
    assert.equal(name, "popstate");
    popstate = fn;
    handlers++;
  } };
  t.after(() => {
    globalThis.document = originalDocument;
    globalThis.window = originalWindow;
  });
  initializeStudyExplorer(root);
  const initialQueries = queries;
  const initialHandlers = handlers;
  assert.ok(initialQueries > 0 && initialHandlers > 0);
  assert.deepEqual(rows.map(row => row.hidden), [false, false], "all-selected categories include untagged studies");
  initializeStudyExplorer(root);
  assert.equal(queries, initialQueries);
  assert.equal(handlers, initialHandlers);
  assert.equal(root["data-study-view"], false);
  assert.ok(view.get(".study-view-nav").hidden);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  assert.equal(attrs["aria-pressed"], "true");
  chipHandlers.click();
  t.mock.timers.tick(220);
  assert.equal(attrs["aria-pressed"], "true", "empty selection effectively selects every device");
  assert.equal(active, true);
  for (let i = 0; i < 2; i++) {
    wordHandlers.click();
    t.mock.timers.tick(220);
    assert.equal(wordAttrs["aria-pressed"], "true");
    assert.deepEqual(rows.map(row => row.hidden), [false, false]);
  }
  for (const name of ["unknown", ""]) {
    url.searchParams.set("study", name);
    popstate();
    assert.equal(root["data-study-view"], true);
    assert.equal(view.get(".study-view-nav").hidden, false);
    assert.equal(view.get(".study-view-empty").hidden, false);
    assert.equal(view.get(".study-view-back").href, "https://example.org/explore.html?tab=table#catalog");
    assert.equal(root.dataset.selectedStudy, "");
  }
  url.searchParams.delete("study");
  popstate();
  assert.equal(root["data-study-view"], false);
  assert.ok(view.get(".study-view-nav").hidden);
  assert.ok(view.get(".study-view-empty").hidden);
});

test("Python and bash highlighting keeps untrusted snippets literal", () => {
  const code = {
    ownerDocument: {
      createTextNode(text) { return { textContent: text }; },
      createElement() { return { className: "", textContent: "" }; },
    },
    replaceChildren(...nodes) { this.nodes = nodes; },
    get textContent() { return this.nodes.map(node => node.textContent).join(""); },
    set innerHTML(_) { assert.fail("metadata must not enter an HTML sink"); },
  };
  const python = 'import neuralset as ns\nname="</script><img src=x onerror=alert(1)>"\n# <svg onload=alert(2)>\n';
  highlightSnippet(code, python, "python");
  assert.equal(code.textContent, python);
  assert.deepEqual(new Set(code.nodes.filter(node => node.className).map(node => node.className)),
    new Set(["token keyword", "token string", "token comment"]));
  const bash = "pip install 'example>=1'\n# <script>alert('&')</script>\n";
  highlightSnippet(code, bash, "bash");
  assert.equal(code.textContent, bash);
});
