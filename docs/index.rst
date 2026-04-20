.. toctree::
   :hidden:

   neuralset/index
   neuraltrain/index
   neuralfetch/index

.. raw:: html

   <div class="hero-welcome-banner">
     <i class="fas fa-brain banner-brain-icon"></i>
     <span class="banner-text">neuroai</span>
   </div>

   <div class="hero-section-root">
     <h1 class="hero-tagline-main">
       Simple. Fast. Robust. Scalable.
     </h1>
     <p class="hero-subtitle">
       The Python suite for brain data across all modalities — from raw recordings to state-of-the-art decoding.
     </p>
     <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
       <div style="display: inline-flex; align-items: center; gap: 0.6rem; position: relative;">
         <pre id="hero-pip-install" style="display: inline-block; background: var(--color-background-secondary); padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid var(--color-border); font-family: var(--font-stack--monospace); font-size: 0.9em; margin: 0; color: var(--color-foreground-primary);">pip install neuralset neuralfetch</pre>
         <button class="copy-btn" id="hero-copy-btn" title="Copy install command" style="position: static; flex-shrink: 0;"><i class="fas fa-copy"></i></button>
       </div>
     </div>
     <script>
     (function() {
       var btn = document.getElementById('hero-copy-btn');
       if (!btn) return;
       btn.addEventListener('click', function() {
         navigator.clipboard.writeText('pip install neuralset neuralfetch').then(function() {
           btn.innerHTML = 'Copied ⚡🧠';
           btn.classList.add('copied');
           setTimeout(function() {
             btn.innerHTML = '<i class="fas fa-copy"></i>';
             btn.classList.remove('copied');
           }, 1500);
         });
       });
     })();
     </script>
     <p style="margin-top: 1rem; font-size: 0.9em;">
       New here? Start with <a href="neuralset/index.html" style="color: var(--color-brand-primary); font-weight: 600; text-decoration: none;">neuralset</a> — the core data pipeline.
     </p>
   </div>

   <!-- Packages as card layout -->
   <div class="packages-cards-container">
     <p style="text-align: center; color: var(--color-foreground-muted); margin-bottom: 2rem; font-size: 0.95em;">
       <strong>Each pipeline step maps to a Python package</strong> — neuralfetch handles data discovery,
       neuralset loads & transforms, and neuraltrain trains models.
     </p>
     <div class="packages-cards">
       <!-- Neuralfetch -->
       <div class="package-card-wrapper">
        <a href="neuralfetch/index.html" class="package-card">
          <div class="package-card-icon"><i class="fas fa-download"></i></div>
          <div class="package-card-name">neuralfetch</div>
           <div class="package-card-tagline">Access the world's curated brain datasets.</div>
         </a>
       </div>

       <!-- Neuralset (highlighted as recommended starting point) -->
       <div class="package-card-wrapper">
         <a href="neuralset/index.html" class="package-card package-card--highlight">
           <div class="package-card-icon"><i class="fas fa-database"></i></div>
           <div class="package-card-name">neuralset</div>
           <div class="package-card-tagline">Turn raw brain data into AI-ready features.</div>
         </a>
       </div>

       <!-- Neuraltrain -->
       <div class="package-card-wrapper">
         <a href="neuraltrain/index.html" class="package-card">
           <div class="package-card-icon"><i class="fas fa-dumbbell"></i></div>
           <div class="package-card-name">neuraltrain</div>
           <div class="package-card-tagline">Deep learning for the brain, supercharged.</div>
         </a>
       </div>

       <!-- exca -->
       <div class="package-card-wrapper">
         <a href="https://facebookresearch.github.io/exca/" class="package-card" target="_blank">
           <div class="package-card-icon"><i class="fas fa-cog"></i></div>
           <div class="package-card-name">exca</div>
           <div class="package-card-tagline">Caching & remote compute infrastructure.</div>
         </a>
       </div>
    </div>
  </div>

----

Citation
--------

.. code-block:: text

   @misc{king2026neuralset,
     title  = {NeuralSet: A High-Performing Python Package for Neuro-AI},
     author = {King, Jean-R{\'e}mi and Bel, Corentin and Evanson, Linnea
               and Gadonneix, Julien and Houhamdi, Sophia and L{\'e}vy, Jarod
               and Raugel, Josephine and Santos Revilla, Andrea
               and Zhang, Mingfang and Bonnaire, Julie and Caucheteux, Charlotte
               and D{\'e}fossez, Alexandre and Desbordes, Th{\'e}o
               and Diego-Sim{\'o}n, Pablo and Khanna, Shubh and Millet, Juliette
               and Orhan, Pierre and Panchavati, Saarang and Ratouchniak, Antoine
               and Thual, Alexis and Brooks, Teon L. and Begany, Katelyn
               and Benchetrit, Yohann and Careil, Marl{\`e}ne and Banville, Hubert
               and d'Ascoli, St{\'e}phane and Dahan, Simon and Rapin, J{\'e}r{\'e}my},
     year   = {2026},
     url    = {https://kingjr.github.io/files/neuralset.pdf},
     note   = {Preprint; URL will be updated when the paper lands on arXiv}
   }

.. raw:: html

   <div class="metadata-panel">
     <p style="font-size: 0.85rem; color: var(--color-foreground-muted); margin: 0;">
       Copyright © Meta Platforms, Inc. and affiliates |
       Made with <a href="https://www.sphinx-doc.org/" style="color: inherit; text-decoration: none;">Sphinx</a> and
       <a href="https://pradyunsg.me" style="color: inherit; text-decoration: none;">@pradyunsg</a>'s
       <a href="https://github.com/pradyunsg/furo" style="color: inherit; text-decoration: none;">Furo</a> |
       <a href="https://github.com/facebookresearch/neuroai" style="color: inherit; text-decoration: none;"><i class="fab fa-github"></i></a>
     </p>
   </div>
