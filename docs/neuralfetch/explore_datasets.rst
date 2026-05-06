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
       { name: 'Allen2022Massive',            modality: 'fmri', subjects: 8,   total_hours: 180, aliases: 'NSD', description: 'Natural Scenes Dataset: 7T BOLD fMRI from 8 participants viewing 73,000+ images', sample: 'Allen2022MassiveSample' },
       { name: 'Bel2026PetitListen',      modality: 'meg',  subjects: 58,  total_hours: 87,  aliases: 'LPP-Listen', description: 'MEG from 58 participants listening to Le Petit Prince in French', sample: 'Bel2026PetitListenSample' },
       { name: 'Bel2026PetitRead',        modality: 'meg',  subjects: 58,  total_hours: 87,  aliases: 'LPP-Read', description: 'MEG from 58 participants reading Le Petit Prince in French', sample: 'Bel2026PetitReadSample' },
       { name: 'Grootswagers2022Human',   modality: 'eeg',  subjects: 50,  total_hours: 95,  aliases: 'THINGS-EEG2', description: 'THINGS-EEG2: EEG from 50 participants in a rapid image-stream paradigm (22,248 objects)', sample: 'Grootswagers2022HumanSample' },
       { name: 'Li2022Petit',             modality: 'fmri', subjects: 112, total_hours: 175, aliases: 'LPP-fMRI', description: 'Le Petit Prince fMRI Corpus: 3T fMRI from 112 participants listening to a naturalistic story', sample: 'Li2022PetitSample' },
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
       { name: 'Albrecht2019Increased',   modality: 'eeg',  subjects: 77,   total_hours: null, aliases: '',           description: 'EEG from 77 participants (46 controls, 31 schizophrenia) on a reinforcement-learning Simon task', sample: null },
       { name: 'Brennan2019Hierarchical', modality: 'eeg',  subjects: 33,   total_hours: null, aliases: '',           description: 'EEG from 33 participants listening to 12 min of the Alice in Wonderland audiobook in English', sample: null },
       { name: 'Chen2023Large',           modality: 'eeg',  subjects: 123,  total_hours: null, aliases: 'FACED',      description: 'EEG from 123 participants viewing 28 video clips across 9 emotion categories and 3 valence levels', sample: null },
       { name: 'Dan2023Bids',             modality: 'eeg',  subjects: 23,   total_hours: null, aliases: 'CHB-MIT',    description: 'Long-term EEG seizure monitoring from 23 pediatric participants with intractable seizures', sample: null },
       { name: 'Ghassemi2018You',         modality: 'eeg',  subjects: 1983, total_hours: null, aliases: 'CinC2018',   description: 'Polysomnographic EEG from 1,983 clinical participants for sleep-arousal and stage detection', sample: null },
       { name: 'Hamid2020Tuar',           modality: 'eeg',  subjects: 213,  total_hours: null, aliases: 'TUH-EEG (TUAR)', description: 'TUH EEG subset with artifact event annotations across 213 clinical patients', sample: null },
       { name: 'Harati2015Tuev',          modality: 'eeg',  subjects: 370,  total_hours: null, aliases: 'TUH-EEG (TUEV)', description: 'TUH EEG subset with epileptiform and artifact event annotations across 370 clinical patients', sample: null },
       { name: 'HaratiAbhishaike2015Tuev',modality: 'eeg',  subjects: 370,  total_hours: null, aliases: 'TUH-EEG (TUEV)', description: 'TUH EEG subset with per-second epileptiform-activity and artifact annotations across 370 subjects', sample: null },
       { name: 'Hinss2023Open',           modality: 'eeg',  subjects: 29,   total_hours: 100,  aliases: 'COG-BCI',    description: 'EEG from 29 participants over 3 sessions performing cognitive tasks (PVT, Flanker, N-back, MATB)', sample: null },
       { name: 'Hollenstein2018Zuco',     modality: 'eeg',  subjects: 12,   total_hours: null, aliases: 'ZuCo',       description: 'Simultaneous EEG and eye-tracking from 12 participants reading natural English sentences', sample: null },
       { name: 'Kemp2000Analysis',        modality: 'eeg',  subjects: 78,   total_hours: null, aliases: 'Sleep-EDF',  description: 'Two-night ambulatory polysomnographic EEG from 78 participants studying aging effects on sleep', sample: null },
       { name: 'Kueper2024Eeg',           modality: 'eeg',  subjects: 8,    total_hours: null, aliases: 'IntEr-HRI',  description: 'EEG from 8 participants detecting errors during orthosis-assisted elbow flexion / extension', sample: null },
       { name: 'Liu2024Eeg2video',        modality: 'eeg',  subjects: 20,   total_hours: null, aliases: 'SEED-DV',    description: 'EEG from 20 participants viewing 1,400 two-second video clips across 40 visual concepts', sample: null },
       { name: 'Lopez2017Tuab',           modality: 'eeg',  subjects: 2329, total_hours: null, aliases: 'TUH-EEG (TUAB)', description: 'TUH EEG subset with normal / abnormal recording labels across 2,329 clinical patients', sample: null },
       { name: 'Miltiadous2023Dice',      modality: 'eeg',  subjects: 88,   total_hours: null, aliases: '',           description: "Resting-state eyes-closed EEG from 88 participants (Alzheimer's, FTD, healthy controls)", sample: null },
       { name: 'Mumtaz2018Machine',       modality: 'eeg',  subjects: 64,   total_hours: null, aliases: '',           description: 'EEG from 34 MDD patients and 30 healthy controls across rest and P300 oddball conditions', sample: null },
       { name: 'Nieuwland2018Large',      modality: 'eeg',  subjects: 295,  total_hours: null, aliases: '',           description: 'Multi-site EEG from 295 participants reading sentences via RSVP to test linguistic predictions', sample: null },
       { name: 'Obeid2016Tueg',           modality: 'eeg',  subjects: 14987,total_hours: null, aliases: 'TUH-EEG (TUEG)', description: 'Full TUH EEG clinical superset: 14,987 subjects, ~70k recordings, no label annotations', sample: null },
       { name: 'Schalk2004Bci',           modality: 'eeg',  subjects: 109,  total_hours: null, aliases: 'EEGMMIDB',   description: 'EEG from 109 participants performing motor-execution and motor-imagery tasks (BCI2000)', sample: null },
       { name: 'Shah2018Tusz',            modality: 'eeg',  subjects: 675,  total_hours: null, aliases: 'TUH-EEG (TUSZ)', description: 'TUH EEG subset with per-channel seizure start / duration annotations across 675 subjects', sample: null },
       { name: 'Shirazi2024Hbn',          modality: 'eeg',  subjects: 3146, total_hours: null, aliases: 'HBN-EEG',    description: 'EEG from 3,146 young participants (ages 5-21) across cognitive tasks and movie watching', sample: null },
       { name: 'Singh2021Timing',         modality: 'eeg',  subjects: 129,  total_hours: null, aliases: '',           description: "EEG during a peak-interval timing task in 83 Parkinson's disease patients and 37 controls", sample: null },
       { name: 'Veloso2017Tuep',          modality: 'eeg',  subjects: 200,  total_hours: null, aliases: 'TUH-EEG (TUEP)', description: 'TUH EEG subset labeled epilepsy vs no-epilepsy across 200 clinical subjects', sample: null },
       { name: 'VonWeltin2017Tusl',       modality: 'eeg',  subjects: 38,   total_hours: null, aliases: 'TUH-EEG (TUSL)', description: 'TUH EEG subset with slowing-event annotations across 38 subjects', sample: null },
       { name: 'Zyma2019Eeg',             modality: 'eeg',  subjects: 35,   total_hours: null, aliases: 'EEGMAT',     description: 'EEG from 35 participants performing a mental-arithmetic (serial-subtraction) task', sample: null },
     ];

     var MODALITIES = ['eeg', 'meg', 'fmri', 'ieeg', 'emg'];
     var studiesData = { all: ALL };
     MODALITIES.forEach(function (m) {
       studiesData[m] = ALL.filter(function (d) { return d.modality === m; });
     });
     ALL.forEach(function (d) {
       if (d.total_hours !== null && d.subjects) {
         d.hrs_per_subject = parseFloat((d.total_hours / d.subjects).toFixed(1));
       } else {
         d.hrs_per_subject = null;
       }
     });

     var EMOJI = { meg: '🔬', fmri: '🔴', eeg: '🧠', ieeg: '⚡', emg: '💪' };
     var sortBy = 'name';
     var sortAsc = true;
     var currentModality = 'all';

     function sortDatasets(list, col) {
       return list.slice().sort(function (a, b) {
         var av = a[col], bv = b[col];
         // Nulls always sort last, regardless of direction.
         if (av === null && bv === null) return 0;
         if (av === null) return 1;
         if (bv === null) return -1;
         if (typeof av === 'string') { av = av.toLowerCase(); bv = bv.toLowerCase(); }
         if (sortAsc) { return av < bv ? -1 : av > bv ? 1 : 0; }
         return av > bv ? -1 : av < bv ? 1 : 0;
       });
     }

     function indicator(col) { return sortBy === col ? (sortAsc ? ' &#8593;' : ' &#8595;') : ''; }

     function fmtNum(v) { return v === null ? '\u2014' : v.toLocaleString(); }
     function fmtRaw(v) { return v === null ? '\u2014' : v; }

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
           + '<td>' + fmtNum(d.subjects) + '</td>'
           + '<td>' + fmtRaw(d.total_hours) + '</td>'
           + '<td>' + fmtRaw(d.hrs_per_subject) + '</td>'
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
