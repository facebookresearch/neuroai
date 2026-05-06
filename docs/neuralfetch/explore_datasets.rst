Explore Brain Datasets
======================

Browse all brain datasets accessible through NeuralFetch. Filter by modality
and click column headers to sort. Datasets with a **▶ sample** badge have
a ready-to-run lightweight version — see :doc:`samples`.

.. raw:: html
   :file: _explore_data.html

.. raw:: html

   <div class="explore-datasets-section">
     <div class="modality-selector">
       <button class="explore-modality-btn active" data-modality="all">📊 All</button>
       <button class="explore-modality-btn" data-modality="eeg">🧠 EEG</button>
       <button class="explore-modality-btn" data-modality="meg">🔬 MEG</button>
       <button class="explore-modality-btn" data-modality="fmri">🔴 fMRI</button>
       <button class="explore-modality-btn" data-modality="ieeg">⚡ iEEG</button>
       <button class="explore-modality-btn" data-modality="emg">💪 EMG</button>
       <button class="explore-modality-btn" data-modality="fnirs">💡 fNIRS</button>
     </div>
     <div id="datasetsTable"></div>
   </div>

   <script>
   (function () {
     var ALL = window.EXPLORE_DATASETS || [];
     var MODALITIES = ['eeg', 'meg', 'fmri', 'ieeg', 'emg', 'fnirs'];
     var studiesData = { all: ALL };
     MODALITIES.forEach(function (m) {
       studiesData[m] = ALL.filter(function (d) { return d.modality === m; });
     });
     ALL.forEach(function (d) {
       d.hrs_per_subject = (d.total_hours !== null && d.subjects)
         ? parseFloat((d.total_hours / d.subjects).toFixed(1)) : null;
       // Collapse whitespace runs so multi-line `description` ClassVars
       // don't break the table layout.
       d.description = (d.description || '').replace(/\s+/g, ' ').trim();
     });

     var EMOJI = { meg: '🔬', fmri: '🔴', eeg: '🧠', ieeg: '⚡', emg: '💪', fnirs: '💡' };
     var sortBy = 'name';
     var sortAsc = true;
     var currentModality = 'all';

     function cmp(a, b) {
       // Nulls always sort last, regardless of direction.
       if (a === null && b === null) return 0;
       if (a === null) return 1;
       if (b === null) return -1;
       if (typeof a === 'string') { a = a.toLowerCase(); b = b.toLowerCase(); }
       if (sortAsc) { return a < b ? -1 : a > b ? 1 : 0; }
       return a > b ? -1 : a < b ? 1 : 0;
     }

     function sortDatasets(list, col) {
       return list.slice().sort(function (x, y) { return cmp(x[col], y[col]); });
     }

     function indicator(col) { return sortBy === col ? (sortAsc ? ' &#8593;' : ' &#8595;') : ''; }
     function fmtNum(v) { return v === null ? '\u2014' : v.toLocaleString(); }
     function fmtRaw(v) { return v === null ? '\u2014' : v; }

     function renderTable(modality) {
       currentModality = modality;
       var list = studiesData[modality] || ALL;
       var el = document.getElementById('datasetsTable');
       if (!list.length) {
         el.innerHTML = '<p style="text-align:center;padding:2rem;color:var(--color-foreground-muted)">No dataset yet for this modality.</p>';
         return;
       }
       var sorted = sortDatasets(list, sortBy);
       var MAX_DESC = 180;
       function escAttr(s) { return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;'); }
       var rows = sorted.map(function (d) {
         var badge = d.sample ? ' <span class="explore-sample-badge">&#9654; sample</span>' : '';
         var desc = d.description || '';
         var descCell = desc.length > MAX_DESC
           ? '<span title="' + escAttr(desc) + '">' + desc.slice(0, MAX_DESC).trimEnd() + '\u2026</span>'
           : desc;
         if (d.aliases) {
           descCell += '<div class="explore-aliases">a.k.a. <em>' + d.aliases + '</em></div>';
         }
         return '<tr>'
           + '<td><strong>' + (EMOJI[d.modality] || '') + ' ' + d.name + '</strong>' + badge + '</td>'
           + '<td>' + d.modality.toUpperCase() + '</td>'
           + '<td>' + fmtNum(d.subjects) + '</td>'
           + '<td>' + fmtRaw(d.total_hours) + '</td>'
           + '<td>' + fmtRaw(d.hrs_per_subject) + '</td>'
           + '<td>' + descCell + '</td>'
           + '</tr>';
       }).join('');
       el.innerHTML = '<table class="datasets-table"><thead><tr>'
         + '<th class="sortable" data-col="name">Dataset' + indicator('name') + '</th>'
         + '<th>Modality</th>'
         + '<th class="sortable" data-col="subjects">Subjects' + indicator('subjects') + '</th>'
         + '<th class="sortable" data-col="total_hours">Hours' + indicator('total_hours') + '</th>'
         + '<th class="sortable" data-col="hrs_per_subject">Hrs/subj' + indicator('hrs_per_subject') + '</th>'
         + '<th>Description</th>'
         + '</tr></thead><tbody>' + rows + '</tbody></table>';
       el.querySelectorAll('.sortable').forEach(function (th) {
         th.addEventListener('click', function () {
           var col = th.dataset.col;
           if (sortBy === col) { sortAsc = !sortAsc; } else { sortBy = col; sortAsc = true; }
           renderTable(currentModality);
         });
       });
     }

     document.querySelectorAll('.explore-modality-btn').forEach(function (btn) {
       btn.addEventListener('click', function () {
         document.querySelectorAll('.explore-modality-btn').forEach(function (b) { b.classList.remove('active'); });
         btn.classList.add('active');
         renderTable(btn.dataset.modality);
       });
     });

     renderTable('all');
   })();
   </script>

   <style>
   .explore-sample-badge {
     display: inline-block;
     margin-left: .35rem;
     font-size: .72rem;
     font-weight: 700;
     padding: .1rem .4rem;
     border-radius: 999px;
     background: color-mix(in srgb, #22c55e 15%, transparent);
     color: color-mix(in srgb, #15803d 70%, var(--color-foreground-primary));
     vertical-align: middle;
   }
   </style>

Download a Dataset
------------------

Pass any study name to ``ns.Study`` to download and load it:

.. code-block:: python

   import neuralset as ns

   study = ns.Study(name="Grootswagers2022Human", path="./data")
   study.download()       # fetch raw files from source repository
   events = study.run()   # returns a tidy events DataFrame

Datasets marked **▶ sample** have a lightweight variant you can run
immediately without downloading the full dataset:

.. code-block:: python

   import neuralset as ns

   study = ns.Study(name="Grootswagers2022HumanSample", path="./data")
   study.download()       # downloads a small subset
   events = study.run()   # returns a tidy events DataFrame

See :doc:`samples` for all sample datasets with ready-to-paste snippets,
or :doc:`reference/reference` for the full API reference.

.. raw:: html

   <div class="page-nav">
     <a href="auto_examples/index.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorials</a>
     <a href="samples.html" class="page-nav-btn">Sample Datasets &rarr;</a>
   </div>
