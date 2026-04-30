Sample Datasets
===============

.. raw:: html

   <p class="landing-badge"><i class="fas fa-flask"></i><span>Sample datasets</span></p>

Try any sample study in minutes — no large download needed.
Each sample is a curated slice of a full dataset, designed to work
on a laptop.

.. raw:: html

   <style>
   /* ── Samples page ──────────────────────────────────────────── */
   .smp { --smp-accent:#448aff; --smp-border:rgba(100,116,139,.2);
          --smp-card:var(--color-background-secondary);
          --smp-muted:var(--color-foreground-secondary);
          --smp-code-bg:#0f172a; --smp-out-bg:#111d2e; }
   .smp-filter { display:flex; gap:.45rem; flex-wrap:wrap; margin:.65rem 0 1.1rem; }
   .smp-filter-btn {
     border:1.5px solid var(--smp-border); cursor:pointer; border-radius:2rem;
     padding:.28rem .85rem; font-size:.83rem; font-weight:700;
     background:var(--color-background-secondary); color:var(--color-foreground-secondary);
     transition:background .13s, border-color .13s, color .13s; }
   .smp-filter-btn.active { background:var(--smp-accent); border-color:var(--smp-accent); color:#fff; }
   .smp-filter-btn:hover:not(.active) { border-color:var(--smp-accent); color:var(--smp-accent); }
   .smp-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px,1fr));
               gap:.8rem; margin:.1rem 0 1.4rem; }
   .smp-card { border:1.5px solid var(--smp-border); border-radius:.9rem;
               padding:.8rem 1rem .7rem; background:var(--smp-card); cursor:pointer;
               transition:border-color .12s, transform .12s, box-shadow .12s; }
   .smp-card:hover { transform:translateY(-2px); box-shadow:0 8px 22px rgba(0,0,0,.1); }
   .smp-card.active { border-color:var(--smp-accent); box-shadow:0 0 0 3px rgba(68,138,255,.14); }
   .smp-card-head { display:flex; align-items:center; gap:.5rem; margin-bottom:.38rem; }
   .smp-icon { font-size:1.25rem; line-height:1; flex-shrink:0; }
   .smp-card-name { font-weight:700; font-size:.88rem; line-height:1.3; word-break:break-word; }
   .smp-badge { display:inline-block; border-radius:999px; padding:.09rem .42rem;
                font-size:.71rem; font-weight:700; margin-right:.22rem;
                background:rgba(68,138,255,.15); color:var(--smp-accent); }
   .smp-card-meta { font-size:.81rem; color:var(--smp-muted); margin:.1rem 0; }
   /* Terminal */
   .smp-terminal { border-radius:.8rem; overflow:hidden; background:var(--smp-code-bg) !important;
                   border:1px solid rgba(148,163,184,.14);
                   box-shadow:0 8px 28px rgba(0,0,0,.28); margin:.1rem 0 .4rem; }
   .smp-term-bar { display:flex; align-items:center; justify-content:space-between;
                   padding:.55rem 1rem; background:rgba(15,23,42,.9);
                   border-bottom:1px solid rgba(148,163,184,.1); }
   .smp-term-dots { display:flex; gap:.38rem; }
   .smp-term-dot { width:.62rem; height:.62rem; border-radius:50%; }
   .smp-term-dot.r { background:#ff5f57; }
   .smp-term-dot.y { background:#ffbd2e; }
   .smp-term-dot.g { background:#28c940; }
   .smp-term-fname { font-size:.74rem; color:rgba(148,163,184,.6); font-family:monospace; }
   .smp-copy-btn { border:1px solid rgba(148,163,184,.22); border-radius:.35rem;
                   padding:.17rem .58rem; font-size:.74rem; background:transparent;
                   color:rgba(148,163,184,.8); cursor:pointer; transition:background .12s; }
   .smp-copy-btn:hover { background:rgba(148,163,184,.12); }
   .smp-term-code { padding:.9rem 1.1rem 0; background:var(--smp-code-bg) !important; }
   #smp-code-pre { margin:0; padding:0;
                   font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                   font-size:.81rem; line-height:1.58; white-space:pre;
                   color:#e2e8f0 !important; background:transparent !important; border:0; }
   .smp-term-out { padding:.65rem 1.1rem .9rem; background:var(--smp-out-bg) !important;
                   border-top:1px solid rgba(148,163,184,.1); }
   .smp-out-label { font-size:.71rem; font-weight:700; letter-spacing:.06em;
                    text-transform:uppercase; color:rgba(148,163,184,.55); margin-bottom:.3rem; }
   #smp-out-pre { margin:0; padding:0;
                  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                  font-size:.78rem; line-height:1.52; white-space:pre;
                  color:#94a3b8 !important; background:transparent !important; border:0; }
   </style>

   <div class="smp">

     <div class="smp-filter">
       <button class="smp-filter-btn active" data-mod="all">All</button>
       <button class="smp-filter-btn" data-mod="Eeg">&#x1F9E0; EEG</button>
       <button class="smp-filter-btn" data-mod="Fmri">&#x1F534; fMRI</button>
       <button class="smp-filter-btn" data-mod="Meg">&#x1F52C; MEG</button>
     </div>

     <div class="smp-grid" id="smp-grid">
       <article class="smp-card active" data-name="Grootswagers2022HumanSample" data-modality="Eeg">
         <div class="smp-card-head">
           <span class="smp-icon">&#x1F9E0;</span>
           <span class="smp-card-name">Grootswagers2022HumanSample</span>
         </div>
         <p class="smp-card-meta"><span class="smp-badge">EEG</span> THINGS-EEG2 &middot; 1 subject</p>
         <p class="smp-card-meta">Rapid image-stream &middot; 22,248 objects &middot; 63 channels</p>
       </article>
       <article class="smp-card" data-name="Li2022PetitSample" data-modality="Fmri">
         <div class="smp-card-head">
           <span class="smp-icon">&#x1F534;</span>
           <span class="smp-card-name">Li2022PetitSample</span>
         </div>
         <p class="smp-card-meta"><span class="smp-badge">fMRI</span> Le Petit Prince &middot; 1 subject, run 1</p>
         <p class="smp-card-meta">Text + audio events aligned</p>
       </article>
       <article class="smp-card" data-name="Allen2022MassiveSample" data-modality="Fmri">
         <div class="smp-card-head">
           <span class="smp-icon">&#x1F534;</span>
           <span class="smp-card-name">Allen2022MassiveSample</span>
         </div>
         <p class="smp-card-meta"><span class="smp-badge">fMRI</span> Natural Scenes Dataset &middot; 4 runs</p>
         <p class="smp-card-meta">AWS-hosted &middot; 7T BOLD &middot; image events</p>
       </article>
       <article class="smp-card" data-name="Bel2026PetitListenSample" data-modality="Meg">
         <div class="smp-card-head">
           <span class="smp-icon">&#x1F52C;</span>
           <span class="smp-card-name">Bel2026PetitListenSample</span>
         </div>
         <p class="smp-card-meta"><span class="smp-badge">MEG</span> Le Petit Prince listening &middot; 1 subject</p>
         <p class="smp-card-meta">Speech + word/sentence events</p>
       </article>
       <article class="smp-card" data-name="Bel2026PetitReadSample" data-modality="Meg">
         <div class="smp-card-head">
           <span class="smp-icon">&#x1F52C;</span>
           <span class="smp-card-name">Bel2026PetitReadSample</span>
         </div>
         <p class="smp-card-meta"><span class="smp-badge">MEG</span> Le Petit Prince reading &middot; 1 subject</p>
         <p class="smp-card-meta">Word + sentence events aligned</p>
       </article>
     </div>

     <div class="smp-terminal">
       <div class="smp-term-bar">
         <div class="smp-term-dots">
           <div class="smp-term-dot r"></div>
           <div class="smp-term-dot y"></div>
           <div class="smp-term-dot g"></div>
         </div>
         <span class="smp-term-fname" id="smp-fname">study_quickstart.py</span>
         <button class="smp-copy-btn" id="smp-copy" type="button">Copy</button>
       </div>
       <div class="smp-term-code">
         <pre id="smp-code-pre"></pre>
       </div>
       <div class="smp-term-out">
         <div class="smp-out-label">&#x25B6; sample output</div>
         <pre id="smp-out-pre"></pre>
       </div>
     </div>

     <p style="font-size:.83rem;color:var(--color-foreground-secondary);margin-top:.7rem;">
       Looking for full analysis pipelines? See the
       <a href="/neuralset/index.html">NeuralSet</a> walkthrough tutorials.
     </p>

   </div>

   <script>
   (function () {
     var TEMPLATE = [
       'import neuralset as ns',
       '',
       'study = ns.Study(name="__NAME__", path="/tmp/studies")',
       '',
       '# Download sample files (a few MB, automatic)',
       'study.download()',
       '',
       '# Build events DataFrame',
       'events = study.run()',
       "print(events[['type', 'start', 'duration']].head(5))",
       "print('\\nEvent counts:', events.type.value_counts().to_dict())",
     ].join('\n');

     var OUTPUTS = {
       'Grootswagers2022HumanSample': [
         '    type   start  duration',
         '0    Eeg     0.0    3035.7',
         '1  Image     0.0       0.5',
         '2  Image     0.6       0.5',
         '3  Image     1.2       0.5',
         '4  Image     1.8       0.5',
         '',
         "Event counts: {'Image': 22248, 'Eeg': 1}",
       ].join('\n'),
       'Li2022PetitSample': [
         '      type   start  duration',
         '0     Fmri     0.0    2520.0',
         '1     Word     0.0       0.3',
         '2     Word     0.4       0.3',
         '3     Word     0.8       0.3',
         '4     Word     1.2       0.3',
         '',
         "Event counts: {'Word': 1442, 'Sentence': 159, 'Fmri': 14}",
       ].join('\n'),
       'Allen2022MassiveSample': [
         '    type   start  duration',
         '0   Fmri     0.0    4800.0',
         '1  Image     0.0       3.0',
         '2  Image     4.0       3.0',
         '3  Image     8.0       3.0',
         '4  Image    12.0       3.0',
         '',
         "Event counts: {'Image': 730, 'Fmri': 4}",
       ].join('\n'),
       'Bel2026PetitListenSample': [
         '       type   start  duration',
         '0       Meg     0.0    3420.0',
         '1      Word     0.0      0.35',
         '2      Word     0.4      0.35',
         '3      Word     0.8      0.35',
         '4      Word     1.2      0.35',
         '',
         "Event counts: {'Word': 1442, 'Sentence': 159, 'Meg': 1}",
       ].join('\n'),
       'Bel2026PetitReadSample': [
         '       type   start  duration',
         '0       Meg     0.0    3180.0',
         '1      Word     0.0      0.40',
         '2      Word     0.5      0.40',
         '3      Word     1.0      0.40',
         '4      Word     1.5      0.40',
         '',
         "Event counts: {'Word': 1376, 'Sentence': 152, 'Meg': 1}",
       ].join('\n'),
     };

     var filterBtns = Array.from(document.querySelectorAll('.smp-filter-btn'));
     var cards      = Array.from(document.querySelectorAll('#smp-grid .smp-card'));
     var codeEl     = document.getElementById('smp-code-pre');
     var outEl      = document.getElementById('smp-out-pre');
     var fnameEl    = document.getElementById('smp-fname');
     var copyBtn    = document.getElementById('smp-copy');
     if (!codeEl || !cards.length) return;

     var selected = 'Grootswagers2022HumanSample';
     var activeMod = 'all';

     function toFilename(n) { return n.toLowerCase().replace(/[^a-z0-9]+/g, '_') + '.py'; }

     function render() {
       var visible = [];
       cards.forEach(function (c) {
         var show = activeMod === 'all' || c.dataset.modality === activeMod;
         c.style.display = show ? '' : 'none';
         if (show) visible.push(c);
       });
       if (!visible.some(function (c) { return c.dataset.name === selected; }) && visible.length) {
         selected = visible[0].dataset.name;
       }
       cards.forEach(function (c) { c.classList.toggle('active', c.dataset.name === selected); });
       if (fnameEl) fnameEl.textContent = toFilename(selected);
       codeEl.textContent = TEMPLATE.replace('__NAME__', selected);
       if (outEl) outEl.textContent = OUTPUTS[selected] || '';
     }

     filterBtns.forEach(function (btn) {
       btn.addEventListener('click', function () {
         filterBtns.forEach(function (b) { b.classList.remove('active'); });
         btn.classList.add('active');
         activeMod = btn.dataset.mod;
         render();
       });
     });
     cards.forEach(function (c) {
       c.addEventListener('click', function () { selected = c.dataset.name; render(); });
     });
     if (copyBtn) {
       copyBtn.addEventListener('click', function () {
         var txt = codeEl.textContent || '';
         if (navigator.clipboard) {
           navigator.clipboard.writeText(txt).catch(function () {});
         } else {
           var ta = document.createElement('textarea');
           ta.value = txt; document.body.appendChild(ta); ta.select();
           document.execCommand('copy'); document.body.removeChild(ta);
         }
         copyBtn.textContent = 'Copied!';
         setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1400);
       });
     }
     render();
   })();
   </script>

.. raw:: html

   <div class="page-nav">
     <a href="explore_datasets.html" class="page-nav-btn page-nav-btn--outline">&larr; Explore Datasets</a>
     <a href="contributing.html" class="page-nav-btn">Contributing: New Studies &rarr;</a>
   </div>
