/**
 * sidebar-nav.js
 * Ensures all top-level package links stay accessible in the Furo sidebar.
 * The CSS already keeps packages visible; this script handles any
 * dynamic edge cases (e.g. Furo JS collapsing non-current items).
 */
(function () {
  'use strict';

  function ensurePackagesVisible() {
    var tree = document.querySelector('.sidebar-tree > ul');
    if (!tree) return;
    // Furo collapses non-current top-level items with max-height tricks.
    // Force all toctree-l1 items to remain visible (display:list-item).
    Array.from(tree.children).forEach(function (li) {
      if (li.classList.contains('toctree-l1')) {
        li.style.display = 'list-item';
      }
    });
  }

  /**
   * Hide API Reference sub-items in the sidebar permanently.
   * API sections have hundreds of entries (classes, functions) that clutter
   * the sidebar.  Remove the expand toggle and sub-list entirely — users
   * can still click the API link to navigate there.
   */
  function collapseApiSubItems() {
    // Find all sidebar links whose text is exactly "API" or "API Reference"
    var sidebarLinks = document.querySelectorAll('.sidebar-tree a.reference');
    sidebarLinks.forEach(function (link) {
      var text = link.textContent.trim();
      if (text !== 'API' && text !== 'API Reference') return;

      var li = link.closest('li');
      if (!li) return;

      // Remove the expand checkbox, label, and child list
      var checkbox = li.querySelector(':scope > input.toctree-checkbox');
      var label = li.querySelector(':scope > label');
      var ul = li.querySelector(':scope > ul');
      if (checkbox) checkbox.remove();
      if (label) label.remove();
      if (ul) ul.remove();
      li.classList.remove('has-children');
    });
  }

  /* ── Add copy buttons to raw <pre><code> blocks (accordion snippets etc.)
     that are NOT already handled by sphinx-copybutton or code-selector.js.
     The button is appended to the <pre> itself so CSS can position it
     absolutely at the top-right of the code box (matching the site-wide
     `button.copybtn` look from sphinx-copybutton). ── */
  var ICON_COPY = '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><title>Copy to clipboard</title><path stroke="none" d="M0 0h24v24H0z" fill="none"/><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"/></svg>';

  function addCopyButtonsToRawBlocks() {
    document.querySelectorAll('pre > code').forEach(function (codeEl) {
      var pre = codeEl.parentElement;
      // Skip if already inside a sphinx-copybutton or code-selector wrapper
      if (pre.parentElement.classList.contains('highlight')) return;
      // .code-block-wrapper blocks are managed by their own inline scripts.
      if (pre.closest('.code-block-wrapper')) return;
      // Skip if a copy button has already been added to this <pre>.
      if (pre.querySelector(':scope > button.copybtn')) return;

      // Anchor the button to the <pre> so it sits inside the code box.
      var cs = window.getComputedStyle(pre);
      if (cs.position === 'static') pre.style.position = 'relative';

      var btn = document.createElement('button');
      btn.className = 'copybtn o-tooltip--left';
      btn.innerHTML = ICON_COPY;
      btn.setAttribute('data-tooltip', 'Copy');
      btn.setAttribute('aria-label', 'Copy to clipboard');
      btn.addEventListener('click', function () {
        var text = codeEl.textContent;
        if (navigator.clipboard) {
          navigator.clipboard.writeText(text);
        } else {
          var ta = document.createElement('textarea');
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        }
        btn.classList.add('success');
        setTimeout(function () { btn.classList.remove('success'); }, 1500);
      });
      pre.appendChild(btn);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      ensurePackagesVisible();
      collapseApiSubItems();
      addCopyButtonsToRawBlocks();
    });
  } else {
    ensurePackagesVisible();
    collapseApiSubItems();
    addCopyButtonsToRawBlocks();
  }
})();
