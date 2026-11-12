// Version switcher and outdated-version banner, injected by publish_docs.py
(() => {
  const script = document.currentScript;
  if (window.self !== window.top) return; // iframed widgets
  const current = script.dataset.version; // "dev", "stable" or "X.Y"
  const root = new URL(script.dataset.root, location.href);
  const page = (location.origin + location.pathname).slice(
    new URL(`${current}/`, root).href.length,
  );

  // same page in `version`, else its index
  const resolve = async (version) => {
    const index = new URL(`${version}/`, root);
    const target = new URL(page, index);
    const found = await fetch(target, { method: "HEAD" }).then((r) => r.ok, () => false);
    return found ? target.href + location.hash : index.href;
  };

  fetch(new URL("versions.json", root))
    .then((r) => r.json())
    .then(({ stable, versions }) => {
      const shown = current === "stable" ? stable : current;
      const style = document.createElement("style");
      style.textContent = `
        .version-switcher {
          display: block; margin: 0 auto 1rem; padding: 0.2rem 0.4rem; font: inherit;
          color: var(--color-foreground-primary); background: var(--color-background-primary);
          border: 1px solid var(--color-background-border); border-radius: 0.2rem;
        }
        .version-banner {
          padding: 0.5rem 1rem; text-align: center;
          background: var(--color-announcement-background, #000000dd);
          color: var(--color-announcement-text, #eeebee);
        }
        .version-banner a { color: inherit; text-decoration: underline; }
      `;
      document.head.append(style);

      const select = document.createElement("select");
      select.className = "version-switcher";
      select.setAttribute("aria-label", "Documentation version");
      for (const v of versions) {
        const label = v === "dev" ? "dev (main)" : v === stable ? `${v} (stable)` : v;
        select.add(new Option(label, v === stable ? "stable" : v, false, v === shown));
      }
      select.onchange = async () => {
        location.href = await resolve(select.value);
      };
      const brand = document.querySelector(".sidebar-brand"); // furo
      brand ? brand.after(select) : document.body.prepend(select);

      if (stable === null || shown === stable) return;
      const banner = document.createElement("div");
      banner.className = "version-banner";
      const what = current === "dev" ? "the development version (main)" : `version ${current}`;
      banner.innerHTML = `You are reading the documentation of ${what}. <a>Go to the stable version (${stable})</a>.`;
      const link = banner.querySelector("a");
      link.href = new URL("stable/", root).href;
      resolve("stable").then((href) => (link.href = href));
      document.body.prepend(banner);
    });
})();
