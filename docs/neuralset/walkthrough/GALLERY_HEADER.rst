.. _walkthrough:

Tutorials: the NeuralSet pipeline
==================================

Each tutorial covers one step of the pipeline — from
loading data through to building a DataLoader — with
code you can run and modify.

.. include:: ../../walkthrough/concepts_overview.rst

.. raw:: html

   <!-- Hide the auto-generated thumbnail grid on this page -->
   <style>.sphx-glr-thumbnails { display: none !important; }</style>

   <div class="pipeline-vertical">
     <!-- Study -->
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-database"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Study</div>
         <div class="pipeline-vcard-desc">Interface to an external dataset. Download, iterate timelines, load events.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-study">Snippet</a>
           <a class="pipeline-sub-link" href="02_studies.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-study">
           <pre><code>study = ns.Study(name="Fake2025Meg",&#10;                 path=ns.CACHE_FOLDER)&#10;events = study.run()&#10;print(f"{len(events)} events, "&#10;      f"{events['subject'].nunique()} subjects")</code></pre>
           <div class="accordion-footer"><a href="02_studies.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <!-- Events -->
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-table"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Events DataFrame</div>
         <div class="pipeline-vcard-desc">Typed DataFrame rows. Neural, stimuli, text &#x2014; everything is an event.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-events">Snippet</a>
           <a class="pipeline-sub-link" href="01_events.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-events">
           <pre><code>from neuralset.events import Event&#10;evt = Event(type="Word", start=1.0,&#10;            duration=0.3, timeline="sub-01")&#10;print(evt)           # pydantic model&#10;print(evt.model_dump())  # dict</code></pre>
           <div class="accordion-footer"><a href="01_events.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <!-- Transforms -->
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-shuffle"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Transforms</div>
         <div class="pipeline-vcard-desc">Modify the events DataFrame: split, chunk, align, add context.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-transforms">Snippet</a>
           <a class="pipeline-sub-link" href="03_transforms.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-transforms">
           <pre><code>import neuralset as ns&#10;transform = ns.events.transforms.AddSentenceToWords()&#10;events = transform(events)&#10;print(events[events.type == "Sentence"].head())</code></pre>
           <div class="accordion-footer"><a href="03_transforms.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <!-- Extractors -->
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-cubes"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Extractors</div>
         <div class="pipeline-vcard-desc">Convert events into tensors. EEG, fMRI, text, images, audio.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-extractors">Snippet</a>
           <a class="pipeline-sub-link" href="04_extractors.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-extractors">
           <pre><code>meg = ns.extractors.MegExtractor(frequency=100.0)&#10;freq = ns.extractors.WordFrequency(language="english")&#10;sample = meg(events, start=0.0, duration=1.0)&#10;print(f"MEG shape: {sample.shape}")</code></pre>
           <div class="accordion-footer"><a href="04_extractors.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
    <!-- Segmenter & Dataset -->
    <div class="pipeline-vcard">
      <div class="pipeline-vcard-icon"><i class="fas fa-scissors"></i></div>
      <div class="pipeline-vcard-body">
        <div class="pipeline-vcard-title">Segmenter &amp; Dataset</div>
        <div class="pipeline-vcard-desc">Time segments around triggers, then a <code>torch.utils.data.Dataset</code> ready for a DataLoader.</div>
        <div class="pipeline-sub-links">
          <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-segmenter">Snippet</a>
          <a class="pipeline-sub-link" href="05_segmenter.html">Tutorial</a>
        </div>
        <div class="pipeline-accordion-panel" id="acc-segmenter">
          <pre><code>meg = ns.extractors.MegExtractor(frequency=100.0)&#10;segmenter = ns.dataloader.Segmenter(&#10;    start=-0.1, duration=0.5,&#10;    trigger_query='type=="Word"',&#10;    extractors=dict(meg=meg),&#10;    drop_incomplete=True)&#10;dataset = segmenter.apply(events)&#10;loader = DataLoader(dataset, batch_size=8,&#10;                    collate_fn=dataset.collate_fn)</code></pre>
          <div class="accordion-footer"><a href="05_segmenter.html">Open full tutorial &#x2192;</a></div>
        </div>
      </div>
    </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <!-- Chains -->
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-link"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Putting it Together</div>
         <div class="pipeline-vcard-desc">Compose full pipelines: Studies + Transforms + Extractors + Segmenter, all in one config.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-chains">Snippet</a>
           <a class="pipeline-sub-link" href="06_chains.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-chains">
           <pre><code>import neuralset as ns&#10;from neuralset.events import transforms&#10;chain = ns.Chain(steps=[&#10;    ns.Study(name="Fake2025Meg",&#10;             path=ns.CACHE_FOLDER),&#10;    transforms.AddSentenceToWords(),&#10;])&#10;events = chain.run()</code></pre>
           <div class="accordion-footer"><a href="06_chains.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
   </div>

.. raw:: html

   <div class="page-nav">
     <a href="../../install.html" class="page-nav-btn page-nav-btn--outline">&larr; Installation</a>
     <a href="01_events.html" class="page-nav-btn">Tutorial 1: Events &rarr;</a>
   </div>
