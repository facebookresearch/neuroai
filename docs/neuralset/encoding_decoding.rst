Encoding & Decoding
===================

Build ML models to **encode** brain responses or **decode** brain activity
from neural signals across modalities. Select your task, device, and
direction below to generate the code.

.. raw:: html

   <div class="admonition note">
     <p class="admonition-title">Note</p>
     <p>
       Datasets on this page are shipping next week.
       For runnable code now, see the <a href="index.html">quickstart</a>.
     </p>
   </div>

.. raw:: html

   <div class="code-selector">

     <div class="selector-compact">
       <label class="selector-item">
         <span class="selector-label">🎯 Task</span>
         <select id="sel-task">
           <option value="language" selected>🗣️ Language</option>
           <option value="image">🖼️ Image</option>
           <option value="video">🎬 Video</option>
           <option value="classification">🏷️ Classification</option>
         </select>
       </label>
       <label class="selector-item">
         <span class="selector-label">📡 Device</span>
         <select id="sel-device">
           <option value="meg" selected>🧲 MEG</option>
           <option value="eeg">⚡ EEG</option>
           <option value="fmri">🧠 fMRI</option>
           <option value="ieeg">🔬 iEEG</option>
           <option value="emg">💪 EMG</option>
         </select>
       </label>
     </div>

     <div class="code-block-wrapper">
       <div class="code-block-label">
         <i class="fas fa-download"></i> <strong>Setup:</strong> run once in your terminal
       </div>
       <pre><code id="code-install" class="language-bash"></code></pre>
     </div>

     <div class="code-block-wrapper">
       <div class="code-block-label">
         <i class="fas fa-database"></i> <strong>Step 1:</strong> Load study data & configure extractors
       </div>
       <pre><code id="code-data" class="language-python"></code></pre>
     </div>

     <div class="code-separator">
       <i class="fas fa-arrow-down"></i> Optional: fit a ridge regression with sklearn <i class="fas fa-arrow-down"></i>
     </div>

     <div class="selector-compact selector-compact--inline">
       <label class="selector-item">
         <span class="selector-label">🎯 Direction</span>
         <select id="sel-direction">
           <option value="decoding" selected>🔍 Decoding (Brain → Stimulus)</option>
           <option value="encoding">📝 Encoding (Stimulus → Brain)</option>
         </select>
       </label>
     </div>

     <div class="code-block-wrapper">
       <div class="code-block-label">
         <i class="fas fa-cogs"></i> <strong>Step 2:</strong> Machine learning pipeline
       </div>
      <pre><code id="code-sklearn" class="language-python"></code></pre>
    </div>

     <noscript>
       <p>This page requires JavaScript to generate code snippets.
       See the <a href="index.html">quickstart</a> for a static example.</p>
     </noscript>

   </div>

.. raw:: html

   <div class="page-nav">
     <a href="caching_and_cluster.html" class="page-nav-btn page-nav-btn--outline">&larr; Caching &amp; Cluster</a>
     <a href="glossary.html" class="page-nav-btn">Glossary &rarr;</a>
   </div>

