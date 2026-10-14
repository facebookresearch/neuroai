NeuralFetch
===========

Goal
----

NeuralFetch connects to **12 public repositories** through a pluggable backend
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
     <div class="hub-center"><i class="fas fa-download"></i> NeuralFetch</div>
     <div class="hub-arrow-down">↓</div>
     <div class="hub-output">
       <i class="fas fa-table"></i>
       NeuralSet <code>Events DataFrame</code>
     </div>
   </div>

----

Quick install
-------------


.. code-block:: bash

   pip install neuralfetch

Installing NeuralFetch automatically registers all curated studies in
NeuralSet's catalog — no extra imports needed. Downloading their data
needs the backends in the ``quickstart`` extra:

.. code-block:: bash

   pip install 'neuralfetch[quickstart]'

.. code-block:: python

   import neuralset as ns

   study = ns.Study(name="Bel2026PetitListenSample", path="/data")  # MEG + speech, OpenNeuro ds007523
   study.download()
   events = study.run()
   print(events[["type", "start", "duration", "subject", "text"]].head())

----

Explore Available Studies
-------------------------

Browse all studies from their declared ``StudyInfo`` metadata, filter by
event type, and click any study for a ready-to-paste snippet and source
link.

.. raw:: html
   :file: _explore_studies.html

----

Tutorials
---------

Each tutorial walks through one building block of the NeuralFetch pipeline.

.. raw:: html

   <div class="pipeline-vertical">

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-search"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Fetch a curated study</div>
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
       <div class="pipeline-vcard-icon"><i class="fas fa-plus-circle"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Create or share a study</div>
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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: NeuralFetch

   install
   Tutorials <auto_examples/index>
   samples
   Contributing: New Studies <contributing>
   API <reference/reference>
