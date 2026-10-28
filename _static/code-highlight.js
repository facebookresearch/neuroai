/* code-highlight.js — Pygments-style syntax highlighting via Prism.js.
   Used by code-builder.js to colourise dynamically rendered Python and
   bash snippets. Prism is vendored under _static/vendor/prism/ and
   loaded ahead of this file by docs/conf.py.

   Prism emits `<span class="token keyword">…</span>` etc.; the
   `.token.*` colour rules in custom.css map those to the same palette
   as Pygments-rendered blocks so the page looks identical to the
   server-rendered Sphinx code.

   Exposes: window.codeHighlight = { python, bash, escapeHtml }       */

(function () {
  if (typeof window === "undefined") return;
  if (window.codeHighlight) return; // idempotent

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function hl(lang) {
    return function (code) {
      var P = window.Prism;
      if (!P || !P.languages || !P.languages[lang]) {
        return escapeHtml(code);
      }
      return P.highlight(code, P.languages[lang], lang);
    };
  }

  window.codeHighlight = {
    python: hl("python"),
    bash: hl("bash"),
    escapeHtml: escapeHtml,
  };
})();
