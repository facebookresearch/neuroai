/* Pipeline accordion: toggle code-snippet panels on pill click.
   Only one panel open at a time (accordion behaviour).              */

document.addEventListener("DOMContentLoaded", function () {
  // Strip leading indent that RST raw:: html forces on <pre> content.
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
    el.textContent = lines.join("\n");
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
