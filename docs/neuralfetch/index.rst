neuralfetch
===========

.. raw:: html

   <p class="landing-badge landing-badge--hero"><i class="fas fa-download"></i><span>neuralfetch</span></p>

**One interface to fetch neuroimaging datasets from OpenNeuro, DANDI, OSF,
HuggingFace, Zenodo, and more — into a unified events DataFrame ready for
neuralset.**

.. code-block:: bash

   pip install neuralfetch

Installing neuralfetch automatically registers all curated studies in
neuralset's catalog — no extra imports needed.

----

Quickstart
----------

Pick a sample dataset and see the full fetch-to-events workflow.

.. raw:: html

   <div class="code-selector" data-quickstart="true">
     <div class="selector-compact">
       <label class="selector-item">
         <span class="selector-label">📋 Dataset</span>
         <select id="sel-nh-preset">
           <option value="listen-meg" selected>🗣️ Language + MEG — Bel2026PetitListenSample</option>
           <option value="images-eeg">🖼️ Images + EEG — Grootswagers2022HumanSample</option>
           <option value="language-fmri">📖 Language + fMRI — Li2022PetitSample</option>
           <option value="images-fmri">🖼️ Images + fMRI — Allen2022MassiveSample</option>
         </select>
       </label>
     </div>
     <div class="code-block-wrapper">
       <div class="code-block-label"><i class="fas fa-download"></i> Download &amp; load events</div>
       <pre><code id="nh-code-data" class="language-python"></code></pre>
     </div>
   </div>

   <script>
   (function () {
     var PRESETS = {
       'listen-meg': [
         'import neuralset as ns',
         '',
         '# MEG + speech — Bel2026PetitListenSample',
         '# (sample: 1 subject, automatic download)',
         'study = ns.Study(name="Bel2026PetitListenSample", path="./data")',
         'study.download()',
         '',
         'events = study.run()',
         'events[["type", "start", "duration", "text"]].head(10)',
         '',
         '# type     start  duration  text',
         '# Meg        0.0     396.0  NaN',
         '# Audio      0.0      42.3  NaN',
         '# Word       1.52      0.22  there',
         '# Word       1.74      0.18  was',
       ].join('\n'),
       'images-eeg': [
         'import neuralset as ns',
         '',
         '# EEG + images — Grootswagers2022HumanSample',
         '# (THINGS-EEG2 sample: 1 subject, OpenNeuro subset)',
         'study = ns.Study(name="Grootswagers2022HumanSample", path="./data")',
         'study.download()',
         '',
         'events = study.run()',
         'events[["type", "start", "duration", "filepath"]].head(10)',
         '',
         '# type   start  duration  filepath',
         '# Eeg      0.0   3035.7  /data/.../sub-01_raw.fif',
         '# Image    0.0      0.5  /data/images/object0001.jpg',
         '# Image    0.6      0.5  /data/images/object0002.jpg',
       ].join('\n'),
       'language-fmri': [
         'import neuralset as ns',
         '',
         '# fMRI + language — Li2022PetitSample',
         '# (Le Petit Prince: 1 subject, run 1)',
         'study = ns.Study(name="Li2022PetitSample", path="./data")',
         'study.download()',
         '',
         'events = study.run()',
         'events[["type", "start", "duration", "text"]].head(10)',
         '',
         '# type    start   duration  text',
         '# Fmri      0.0    1800.0  NaN',
         '# Audio     0.0    1800.0  NaN',
         '# Word      5.4       0.3  Le',
         '# Word      5.7       0.2  Petit',
       ].join('\n'),
       'images-fmri': [
         'import neuralset as ns',
         '',
         '# fMRI + images — Allen2022MassiveSample',
         '# (Natural Scenes Dataset: 7T BOLD, 4 runs)',
         'study = ns.Study(name="Allen2022MassiveSample", path="./data")',
         'study.download()',
         '',
         'events = study.run()',
         'events[["type", "start", "duration", "filepath"]].head(10)',
         '',
         '# type   start  duration  filepath',
         '# Fmri     0.0    7200.0  /data/sub-01_task-nsd_run01.fif',
         '# Image    0.0       3.0  /data/images/nsd_00001.jpg',
         '# Image    3.0       3.0  /data/images/nsd_00002.jpg',
       ].join('\n'),
     };

     var sel = document.getElementById('sel-nh-preset');
     var codeEl = document.getElementById('nh-code-data');
     if (!sel || !codeEl) return;

     var wrapper = codeEl.closest('.code-block-wrapper') || codeEl.parentElement.parentElement;

     function addCopyBtn(code) {
       var existing = wrapper.querySelector('.copy-btn');
       if (existing) existing.remove();
       var btn = document.createElement('button');
       btn.className = 'copy-btn';
       btn.title = 'Copy code';
       btn.innerHTML = '<i class="fas fa-copy"></i>';
       btn.addEventListener('click', function() {
         navigator.clipboard.writeText(code).then(function() {
           btn.innerHTML = 'Copied ⚡\u{1F9E0}';
           btn.classList.add('copied');
           setTimeout(function() {
             btn.innerHTML = '<i class="fas fa-copy"></i>';
             btn.classList.remove('copied');
           }, 1500);
         });
       });
       wrapper.appendChild(btn);
     }

     function update() {
       var code = PRESETS[sel.value] || '';
       if (window._highlightPy) {
         codeEl.innerHTML = window._highlightPy(code);
       } else {
         codeEl.textContent = code;
       }
       addCopyBtn(code);
     }

     sel.addEventListener('change', update);
     // DOMContentLoaded may have already fired; call update() directly if so.
     if (document.readyState === 'loading') {
       document.addEventListener('DOMContentLoaded', function() {
         // Deferred: ensure _highlightPy from pipeline-accordion.js is available
         setTimeout(update, 0);
       });
     } else {
       update();
     }
   })();
   </script>

----

Tutorials
---------

Each tutorial walks through one building block of the neuralfetch pipeline.

.. raw:: html

   <div class="pipeline-vertical">

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-search"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Fetch Your First Study</div>
         <div class="pipeline-vcard-desc">Browse the catalog, download a sample dataset, and preview the events DataFrame.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nh-fetch">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/01_plot_fetch_first_study.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nh-fetch">
           <pre><code>study = ns.Study(name="Grootswagers2022HumanSample",&#10;                    path="./data")&#10;study.download()&#10;events = study.run()&#10;print(events[["type","start","duration"]].head())</code></pre>
           <div class="accordion-footer"><a href="auto_examples/01_plot_fetch_first_study.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-layer-group"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Study Anatomy</div>
         <div class="pipeline-vcard-desc">Timelines, event types, subject metadata, and how to compose studies with transforms.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nh-anatomy">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/02_plot_study_anatomy.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nh-anatomy">
           <pre><code>for tl in study.iter_timelines():&#10;    print(tl)   # {subject: "sub-01", session: "..."}&#10;&#10;events = study.run()&#10;print(events["type"].value_counts())</code></pre>
           <div class="accordion-footer"><a href="auto_examples/02_plot_study_anatomy.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-plus-circle"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Create Your Own Study</div>
         <div class="pipeline-vcard-desc">Wrap any local or remote dataset as a Study subclass and register it in the catalog.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nh-extend">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/03_create_new_study.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nh-extend">
           <pre><code>class MyStudy(studies.Study):
    def iter_timelines(self):
        yield {"subject": "sub-01"}
    def _load_timeline_events(self, tl):
        return pd.DataFrame([...])</code></pre>
           <div class="accordion-footer"><a href="auto_examples/03_create_new_study.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

   </div>

.. raw:: html

   <div class="page-nav">
     <a href="install.html" class="page-nav-btn page-nav-btn--outline">Installation &rarr;</a>
     <a href="auto_examples/index.html" class="page-nav-btn">Go to Tutorials &rarr;</a>
   </div>

----

How It Works
------------

neuralfetch connects to **12 public repositories** through a pluggable backend
system and returns the same tidy events DataFrame regardless of the source.

.. raw:: html

   <div class="hub-diagram">
     <div class="hub-sources">
       <span class="hub-source"><i class="fas fa-database"></i> DANDI</span>
       <span class="hub-source"><i class="fas fa-code-branch"></i> DataLad</span>
       <span class="hub-source"><i class="fas fa-university"></i> Donders</span>
       <span class="hub-source"><i class="fas fa-leaf"></i> Dryad</span>
       <span class="hub-source"><i class="fas fa-wave-square"></i> EEGDash</span>
       <span class="hub-source"><i class="fas fa-chart-bar"></i> Figshare</span>
       <span class="hub-source"><i class="fas fa-robot"></i> HuggingFace</span>
       <span class="hub-source"><i class="fas fa-brain"></i> OpenNeuro</span>
       <span class="hub-source"><i class="fas fa-flask"></i> OSF</span>
       <span class="hub-source"><i class="fas fa-heartbeat"></i> PhysioNet</span>
       <span class="hub-source"><i class="fas fa-key"></i> Synapse</span>
       <span class="hub-source"><i class="fas fa-archive"></i> Zenodo</span>
     </div>
     <div class="hub-arrows">↓ ↓ ↓ ↓ ↓</div>
     <div class="hub-center"><i class="fas fa-download"></i> neuralfetch</div>
     <div class="hub-arrow-down">↓</div>
     <div class="hub-output">
       <i class="fas fa-table"></i>
       neuralset <code>Events DataFrame</code>
     </div>
   </div>

The result is always the same: a tidy DataFrame where each row is an event
(brain recording, word, audio, image, stimulus...) with timing, subject, and
session information — regardless of where the data came from.

.. code-block:: python

   import neuralset as ns

   study = ns.Study(name="Gwilliams2022Neural", path="/data")  # MEG + speech, from OSF
   events = study.run()
   events[["type", "start", "duration", "subject", "text"]].head()

.. code-block:: text

   type   start  duration              subject          text
   Meg      0.0     396.0  Gwilliams2022Neural/A0001           NaN
   Audio    0.0      42.3  Gwilliams2022Neural/A0001           NaN
   Word     1.52     0.22  Gwilliams2022Neural/A0001         there
   Word     1.74     0.18  Gwilliams2022Neural/A0001           was
   Word     1.92     0.08  Gwilliams2022Neural/A0001             a

----

Supported Data Sources
----------------------

neuralfetch downloads datasets from **12 public repositories** through a
pluggable backend system.  Each backend handles authentication, pagination,
and format differences so your code doesn't have to.

.. list-table::
   :widths: 18 34 48
   :header-rows: 1

   * - Backend
     - Repository
     - What's there
   * - ``Dandi``
     - `DANDI Archive <https://dandiarchive.org>`_
     - NWB neurophysiology datasets (EEG, ephys, calcium imaging)
   * - ``Datalad``
     - `DataLad <https://www.datalad.org>`_
     - Git-annex managed large datasets
   * - ``Donders``
     - `Donders Repository <https://data.donders.ru.nl>`_
     - Donders Institute data collections
   * - ``Dryad``
     - `Dryad <https://datadryad.org>`_
     - Curated, open-access research data repository
   * - ``Eegdash``
     - `EEGDash <https://eegdash.com>`_
     - Cloud-hosted EEG datasets with REST API
   * - ``Figshare``
     - `Figshare <https://figshare.com>`_
     - Research data sharing platform
   * - ``Huggingface``
     - `HuggingFace Hub <https://huggingface.co/datasets>`_
     - ML-ready datasets (neural, audio, text)
   * - ``Openneuro``
     - `OpenNeuro <https://openneuro.org>`_
     - 1 000+ BIDS datasets — EEG, MEG, fMRI, iEEG
   * - ``Osf``
     - `Open Science Framework <https://osf.io>`_
     - General-purpose research data hosting
   * - ``Physionet``
     - `PhysioNet <https://physionet.org>`_
     - Physiological signal databases (EEG, ECG, EMG)
   * - ``Synapse``
     - `Synapse <https://www.synapse.org>`_
     - Collaborative research platform (auth required)
   * - ``Zenodo``
     - `Zenodo <https://zenodo.org>`_
     - CERN-backed open-data archive

Each study declares which backend to use.  You never interact with backends
directly — just call ``study.run()`` and the data is downloaded, cached, and
returned as events.

----

The Events DataFrame
--------------------

Every study produces the same output: a **pandas DataFrame** where each row
is a time-stamped event.  Different event types (brain data, words, audio,
stimuli) coexist in the same table, identified by the ``type`` column.

**Core columns** (always present):

.. list-table::
   :widths: 18 14 68
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``type``
     - str
     - Event class: ``Meg``, ``Eeg``, ``Fmri``, ``Word``, ``Audio``, ``Image``, ``Stimulus``, ...
   * - ``start``
     - float
     - Onset time in seconds
   * - ``duration``
     - float
     - Duration in seconds
   * - ``stop``
     - float
     - ``start + duration`` (auto-computed)
   * - ``timeline``
     - str
     - Unique recording identifier
   * - ``subject``
     - str
     - Subject ID (BIDS format: ``StudyName/sub_id``)
   * - ``session``
     - str
     - Session ID (BIDS entity, ``""`` if unused)
   * - ``task``
     - str
     - Task ID (BIDS entity, ``""`` if unused)
   * - ``run``
     - str
     - Run ID (BIDS entity, ``""`` if unused)

**Type-specific columns** are added depending on event type:

- **Brain data** (``Meg``, ``Eeg``, ``Fmri``, ...): ``filepath``, ``frequency``
- **Text** (``Word``, ``Sentence``, ``Phoneme``): ``text``, ``language``, ``modality``, ``context``
- **Audio / Video**: ``filepath``, ``frequency``, ``offset``
- **Categorical** (``Stimulus``, ``Action``, ``SleepStage``, ...): ``code``, ``description``, ``state``

.. code-block:: text

   Example: an MEG study with speech stimuli
   ──────────────────────────────────────────
   type      start  duration  subject           filepath               text     language
   Meg         0.0     300.0  Study/s01         /data/s01_raw.fif      NaN      NaN
   Audio       0.0     300.0  Study/s01         /data/audio.wav        NaN      NaN
   Word        1.50      0.3  Study/s01         NaN                    hello    english
   Word        2.10      0.2  Study/s01         NaN                    world    english
   Sentence    1.50      0.8  Study/s01         NaN                    hello..  english
   Stimulus    5.00      0.1  Study/s01         NaN                    NaN      NaN

This flat representation makes it easy to filter, group, and join events
across modalities using standard pandas operations.

----

Supported Datasets
------------------

neuralfetch provides **interfaces to public datasets** spanning six recording modalities, with new datasets added regularly:

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Modality
     - Device
     - Example studies
   * - MEG
     - ``Meg``
     - Gwilliams2022Neural, Hebart2023ThingsMeg, Armeni2022Sherlock, ...
   * - EEG
     - ``Eeg``
     - Gifford2022Large, Grootswagers2022Human, Hollenstein2018Zuco, ...
   * - fMRI
     - ``Fmri``
     - Lebel2023Natural, Hebart2023ThingsBold, Allen2022Massive, ...
   * - iEEG
     - ``Ieeg``
     - Wang2024Treebank, ...
   * - EMG
     - ``Emg``
     - Sivakumar2024Emg2qwerty, ...
   * - fNIRS
     - ``Fnirs``
     - Luke2021Analysis, ...

Each study provides:

- **Download logic** — automatically fetches from the right repository
- **Timeline iteration** — yields subject × session × run combinations
- **Event parsing** — converts raw annotations into the events DataFrame
- **Validated metadata** — expected shapes, frequencies, event types, subject counts

----

Browse All Datasets
-------------------

Filter and sort all available datasets by modality, subjects, and hours:

.. raw:: html

   <div style="text-align:center;margin:1rem 0 1.5rem;">
     <a href="explore_datasets.html" class="help-button"><i class="fas fa-table"></i>&nbsp; Browse All Datasets</a>
   </div>

Sample datasets are available for immediate use — no large download required:

.. raw:: html

   <div style="text-align:center;margin:1rem 0 1.5rem;">
     <a href="samples.html" class="help-button"><i class="fas fa-flask"></i>&nbsp; View Sample Datasets</a>
   </div>

----

Citation
--------

See the :doc:`neuralset page </neuralset/index>` for the preferred
citation (BibTeX at the bottom).

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: neuralfetch

   install
   Tutorials <auto_examples/index>
   explore_datasets
   samples
   Contributing: New Studies <contributing>
   API <reference/reference>
