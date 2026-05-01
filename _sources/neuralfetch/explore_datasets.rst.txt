Explore Brain Datasets
======================

Browse all brain datasets accessible through NeuralFetch. Filter by modality
and click column headers to sort. Datasets with a **▶ sample** badge have a
ready-to-run lightweight version — see :doc:`samples`.

.. raw:: html

   <div class="explore-datasets-section">
     <div class="modality-selector">
       <button class="explore-modality-btn active" data-modality="all">📊 All</button>
       <button class="explore-modality-btn" data-modality="eeg">🧠 EEG</button>
       <button class="explore-modality-btn" data-modality="meg">🔬 MEG</button>
       <button class="explore-modality-btn" data-modality="fmri">🔴 fMRI</button>
       <button class="explore-modality-btn" data-modality="ieeg">⚡ iEEG</button>
       <button class="explore-modality-btn" data-modality="emg">💪 EMG</button>
     </div>
     <div id="datasetsTable"></div>
   </div>

   <script>
   (function () {
     var ALL = [
       // Phase 1 — sample datasets available
       { name: 'Allen2022Massive',            modality: 'fmri', subjects: 8,   total_hours: 180, aliases: 'NSD', description: 'Natural Scenes Dataset: 7T BOLD fMRI from 8 participants viewing 73,000+ images', sample: 'Allen2022MassiveSample' },
       { name: 'Bel2026PetitListen',      modality: 'meg',  subjects: 58,  total_hours: 87,  aliases: 'LPP-Listen', description: 'MEG from 58 participants listening to Le Petit Prince in French', sample: 'Bel2026PetitListenSample' },
       { name: 'Bel2026PetitRead',        modality: 'meg',  subjects: 58,  total_hours: 87,  aliases: 'LPP-Read', description: 'MEG from 58 participants reading Le Petit Prince in French', sample: 'Bel2026PetitReadSample' },
       { name: 'Grootswagers2022Human',   modality: 'eeg',  subjects: 50,  total_hours: 95,  aliases: 'THINGS-EEG2', description: 'THINGS-EEG2: EEG from 50 participants in a rapid image-stream paradigm (22,248 objects)', sample: 'Grootswagers2022HumanSample' },
       { name: 'Li2022Petit',             modality: 'fmri', subjects: 112, total_hours: 175, aliases: 'LPP-fMRI', description: 'Le Petit Prince fMRI Corpus: 3T fMRI from 112 participants listening to a naturalistic story', sample: 'Li2022PetitSample' },
       // Phase 2 — remaining release studies
       { name: 'Armeni2022Sherlock',      modality: 'meg',  subjects: 3,   total_hours: 30,  aliases: '', description: 'MEG from 3 participants listening to 10 hours of Sherlock Holmes stories', sample: null },
       { name: 'Chang2019Bold5000',       modality: 'fmri', subjects: 4,   total_hours: 20,  aliases: 'BOLD5000', description: 'BOLD5000: 3T fMRI from 4 participants viewing 5,000 diverse natural images', sample: null },
       { name: 'Gifford2022Large',        modality: 'eeg',  subjects: 10,  total_hours: 80,  aliases: 'THINGS-EEG1', description: 'THINGS-EEG1: EEG from 10 participants viewing 22,248 object images in a rapid RSVP paradigm', sample: null },
       { name: 'Gifford2025Algonauts',    modality: 'fmri', subjects: 4,   total_hours: 80,  aliases: 'Algonauts 2025', description: 'Algonauts 2025: fMRI responses to videos from the Courtois NeuroMod dataset', sample: null },
       { name: 'Gwilliams2022Neural',     modality: 'meg',  subjects: 27,  total_hours: 54,  aliases: 'MASC', description: 'MEG from 27 participants listening to 4 naturalistic stories (MASC corpus)', sample: null },
       { name: 'Hebart2023ThingsBold',    modality: 'fmri', subjects: 3,   total_hours: 36,  aliases: 'THINGS-fMRI1', description: 'THINGS-fMRI: 3T BOLD from 3 participants viewing 1,854 everyday object images', sample: null },
       { name: 'Hebart2023ThingsMeg',     modality: 'meg',  subjects: 4,   total_hours: 40,  aliases: 'THINGS-MEG', description: 'THINGS-MEG: MEG from 4 participants viewing 22,248 object images', sample: null },
       { name: 'Lahner2024Modeling',      modality: 'fmri', subjects: 10,  total_hours: 15,  aliases: 'BOLD Moments', description: 'BOLD Moments: 3T fMRI from 10 participants viewing 1,102 brief naturalistic video clips', sample: null },
       { name: 'Lebel2023Natural',        modality: 'fmri', subjects: 8,   total_hours: 16,  aliases: '', description: 'Natural Language fMRI: 3T fMRI from 8 participants listening to narrative podcast stories', sample: null },
       { name: 'Nastase2021Narratives',   modality: 'fmri', subjects: 321, total_hours: 450, aliases: 'Narratives', description: 'Narratives: large-scale fMRI from 321 participants across 27 diverse spoken-story stimuli', sample: null },
       { name: 'Ozdogan2025LibriBrain',   modality: 'meg',  subjects: 1,   total_hours: 50,  aliases: 'LibriBrain', description: 'LibriBrain: MEG from 1 participant listening to 50 hours of naturalistic audiobooks', sample: null },
       { name: 'Shen2019Deep',            modality: 'fmri', subjects: 3,   total_hours: 44,  aliases: 'Deep Image Reconstruction, DeepRecon', description: 'Deep Image Reconstruction: 3T fMRI from 3 participants viewing natural images and imagery', sample: null },
       { name: 'Sivakumar2024Emg2qwerty', modality: 'emg',  subjects: 108, total_hours: 216, aliases: 'emg2qwerty', description: 'EMG wristband typing data from 108 participants (both wrists)', sample: null },
       { name: 'Wang2024Treebank',        modality: 'ieeg', subjects: 10,  total_hours: 30,  aliases: '', description: 'sEEG from 10 participants watching movies while narrating (syntax treebank)', sample: null },
       { name: 'Zhou2023Large',           modality: 'fmri', subjects: 30,  total_hours: 30,  aliases: 'HAD', description: 'Human Action Dataset (HAD): 3T fMRI from 30 participants viewing naturalistic action clips', sample: null },
     ];

     var MODALITIES = ['eeg', 'meg', 'fmri', 'ieeg', 'emg'];
     var studiesData = { all: ALL };
     MODALITIES.forEach(function (m) {
       studiesData[m] = ALL.filter(function (d) { return d.modality === m; });
     });
     ALL.forEach(function (d) {
       d.hrs_per_subject = parseFloat((d.total_hours / d.subjects).toFixed(1));
     });

     var EMOJI = { meg: '🔬', fmri: '🔴', eeg: '🧠', ieeg: '⚡', emg: '💪' };
     var sortBy = 'name';
     var sortAsc = true;
     var currentModality = 'all';

     function sortDatasets(list, col) {
       return list.slice().sort(function (a, b) {
         var av = a[col], bv = b[col];
         if (typeof av === 'string') { av = av.toLowerCase(); bv = bv.toLowerCase(); }
         if (sortAsc) { return av < bv ? -1 : av > bv ? 1 : 0; }
         return av > bv ? -1 : av < bv ? 1 : 0;
       });
     }

     function indicator(col) { return sortBy === col ? (sortAsc ? ' &#8593;' : ' &#8595;') : ''; }

     function renderTable(modality) {
       currentModality = modality;
       var list = studiesData[modality] || ALL;
       var el = document.getElementById('datasetsTable');
       if (!list.length) {
         el.innerHTML = '<p style="text-align:center;padding:2rem;color:var(--color-foreground-muted)">No datasets for this modality.</p>';
         return;
       }
       var sorted = sortDatasets(list, sortBy);
       var rows = sorted.map(function (d) {
         var badge = d.sample ? ' <span class="explore-sample-badge">&#9654; sample</span>' : '';
         var aliasCell = d.aliases ? '<em>' + d.aliases + '</em>' : '';
         return '<tr>'
           + '<td><strong>' + (EMOJI[d.modality] || '') + ' ' + d.name + '</strong>' + badge + '</td>'
           + '<td>' + aliasCell + '</td>'
           + '<td>' + d.modality.toUpperCase() + '</td>'
           + '<td>' + d.subjects.toLocaleString() + '</td>'
           + '<td>' + d.total_hours + '</td>'
           + '<td>' + d.hrs_per_subject + '</td>'
           + '<td>' + d.description + '</td>'
           + '</tr>';
       }).join('');
       el.innerHTML = '<table class="datasets-table"><thead><tr>'
         + '<th class="sortable" data-col="name">Dataset' + indicator('name') + '</th>'
         + '<th>Aliases</th>'
         + '<th>Modality</th>'
         + '<th class="sortable" data-col="subjects">Subjects' + indicator('subjects') + '</th>'
         + '<th class="sortable" data-col="total_hours">Total hours' + indicator('total_hours') + '</th>'
         + '<th class="sortable" data-col="hrs_per_subject">Hrs / subject' + indicator('hrs_per_subject') + '</th>'
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
