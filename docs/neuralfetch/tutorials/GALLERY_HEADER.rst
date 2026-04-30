Tutorials
=========

Three tutorials cover the full NeuralFetch workflow — from discovering
public datasets to fetching your first study and building your own.

.. raw:: html

   <style>.sphx-glr-thumbnails { display: none !important; }</style>

   <div class="pipeline-vertical">

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-search"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">1 · Fetch Your First Study</div>
         <div class="pipeline-vcard-desc">Browse the catalog, load a sample, and explore the events DataFrame.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nf-gfetch">Snippet</a>
           <a class="pipeline-sub-link" href="01_plot_fetch_first_study.html">Tutorial &#x2192;</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nf-gfetch">
           <pre><code>import neuralset as ns&#10;&#10;study = ns.Study(name="Grootswagers2022HumanSample",&#10;                 path="./data")&#10;study.download()&#10;events = study.run()&#10;print(events[["type","start","duration"]].head())</code></pre>
           <div class="accordion-footer"><a href="01_plot_fetch_first_study.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-layer-group"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">2 · Anatomy of a Study</div>
         <div class="pipeline-vcard-desc">Timelines, event types, subject metadata, and composing studies with transforms.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nf-ganatomy">Snippet</a>
           <a class="pipeline-sub-link" href="02_plot_study_anatomy.html">Tutorial &#x2192;</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nf-ganatomy">
           <pre><code>for tl in study.iter_timelines():&#10;    print(tl)&#10;&#10;events = study.run()&#10;print(events["type"].value_counts())</code></pre>
           <div class="accordion-footer"><a href="02_plot_study_anatomy.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-plus-circle"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">3 · Create Your Own Study</div>
         <div class="pipeline-vcard-desc">Wrap any local or remote dataset as a Study subclass registered in the catalog.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nf-gcreate">Snippet</a>
           <a class="pipeline-sub-link" href="03_create_new_study.html">Tutorial &#x2192;</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nf-gcreate">
           <pre><code>class MyStudy(studies.Study):&#10;    def iter_timelines(self):&#10;        yield {"subject": "sub-01"}&#10;    def _load_timeline_events(self, tl):&#10;        return pd.DataFrame([...])</code></pre>
           <div class="accordion-footer"><a href="03_create_new_study.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

   </div>

.. raw:: html

   <div class="page-nav">
     <a href="../install.html" class="page-nav-btn page-nav-btn--outline">&larr; Installation</a>
     <a href="01_plot_fetch_first_study.html" class="page-nav-btn">Tutorial 1: Fetch Your First Study &rarr;</a>
   </div>

