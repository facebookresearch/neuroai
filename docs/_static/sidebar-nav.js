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
     that are NOT already handled by sphinx-copybutton or code-selector.js ── */
  function addCopyButtonsToRawBlocks() {
    document.querySelectorAll('pre > code').forEach(function (codeEl) {
      var pre = codeEl.parentElement;
      // Skip if already inside a sphinx-copybutton or code-selector wrapper
      if (pre.parentElement.classList.contains('highlight')) return;
      if (pre.parentElement.querySelector('.copy-btn')) return;
      if (pre.parentElement.querySelector('.copybtn')) return;

      // Ensure the wrapper is position-relative
      var wrapper = pre.parentElement;
      var cs = window.getComputedStyle(wrapper);
      if (cs.position === 'static') wrapper.style.position = 'relative';

      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.innerHTML = '<i class="fas fa-copy"></i>';
      btn.title = 'Copy code';
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
        var orig = btn.innerHTML;
        btn.innerHTML = 'Copied ⚡🧠';
        btn.classList.add('copied');
        setTimeout(function () { btn.innerHTML = orig; btn.classList.remove('copied'); }, 1500);
      });
      wrapper.appendChild(btn);
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
