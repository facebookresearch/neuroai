/* Pipeline accordion: toggle code-snippet panels on pill click.
   Only one panel open at a time (accordion behaviour).              */

document.addEventListener("DOMContentLoaded", function () {

  // ── Minimal Python highlighter (shared logic with code-selector.js) ──
  function _esc(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function _tok(cls,t) { return '<span class="'+cls+'">'+_esc(t)+'</span>'; }
  var _KW = {"import":1,"from":1,"as":1,"def":1,"class":1,"if":1,"elif":1,"else":1,"for":1,"in":1,"while":1,"try":1,"except":1,"finally":1,"with":1,"return":1,"yield":1,"lambda":1,"and":1,"or":1,"not":1,"is":1,"None":1,"True":1,"False":1,"break":1,"continue":1,"pass":1};
  var _BI = {"print":1,"len":1,"range":1,"enumerate":1,"zip":1,"dict":1,"list":1,"tuple":1,"set":1,"str":1,"int":1,"float":1,"bool":1,"type":1};
  function highlightPy(code) {
    var i=0,n=code.length,o="";
    while(i<n){var c=code[i];
      if(c==="#"){var s=i;while(i<n&&code[i]!=="\n")i++;o+=_tok("highlight-comment",code.slice(s,i));continue;}
      if(c==='"'||c==="'"){var q=c,s=i;i++;while(i<n){if(code[i]==="\\"){i+=2;continue;}if(code[i]===q){i++;break;}i++;}o+=_tok("highlight-string",code.slice(s,i));continue;}
      if(/[A-Za-z_]/.test(c)){var s=i;i++;while(i<n&&/[A-Za-z0-9_]/.test(code[i]))i++;var w=code.slice(s,i);if(_KW[w]){o+=_tok("highlight-keyword",w);}else if(_BI[w]){o+=_tok("highlight-builtin",w);}else{var j=i;while(j<n&&/\s/.test(code[j]))j++;o+=(j<n&&code[j]==="(")?_tok("highlight-function",w):_esc(w);}continue;}
      if(/[0-9]/.test(c)||(c==="."&&i+1<n&&/[0-9]/.test(code[i+1]))){var s=i;i++;while(i<n&&/[0-9.]/.test(code[i]))i++;o+=_tok("highlight-number",code.slice(s,i));continue;}
      o+=_esc(c);i++;
    }
    return o;
  }

  // Expose globally so inline scripts (e.g. neuralfetch) can reuse it
  window._highlightPy = highlightPy;

  // Strip leading indent that RST raw:: html forces on <pre> content,
  // then apply syntax highlighting.
  document.querySelectorAll(".pipeline-accordion-panel pre code").forEach(function (el) {
    var lines = el.textContent.split("\n");
    // remove empty first/last lines
    while (lines.length && lines[0].trim() === "") lines.shift();
    while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
    // detect common leading spaces
    var min = Infinity;
    lines.forEach(function (l) { if (l.trim()) min = Math.min(min, l.search(/\S/)); });
    if (min && min < Infinity) {
      lines = lines.map(function (l) { return l.substring(min); });
    }
    el.innerHTML = highlightPy(lines.join("\n"));
  });

  // Accordion toggle
  document.querySelectorAll(".pipeline-accordion-toggle").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      var targetId = btn.getAttribute("data-target");
      var panel = document.getElementById(targetId);
      if (!panel) return;

      var isOpen = panel.classList.contains("open");

      // close every open panel first (accordion)
      document.querySelectorAll(".pipeline-accordion-panel.open").forEach(function (p) {
        p.classList.remove("open");
        p.style.maxHeight = null;
      });
      document.querySelectorAll(".pipeline-accordion-toggle.active").forEach(function (b) {
        b.classList.remove("active");
      });

      // if it was closed, open it
      if (!isOpen) {
        panel.classList.add("open");
        panel.style.maxHeight = panel.scrollHeight + "px";
        btn.classList.add("active");
      }
    });
  });
});
