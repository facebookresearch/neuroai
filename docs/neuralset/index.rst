neuralset
=========

.. raw:: html

   <p class="landing-badge landing-badge--hero"><i class="fas fa-database"></i><span>neuralset</span></p>

**neuralset** turns raw neural recordings and stimuli into PyTorch-ready
datasets. Define a study, load events into a flat DataFrame, apply
transforms, extract features with configurable extractors, and segment
everything into batches — all lazy, typed with pydantic, and cacheable.

.. code-block:: bash

   pip install neuralset

To follow the tutorials and load curated datasets, add tutorial extras
and :doc:`neuralfetch </neuralfetch/index>`:

.. code-block:: bash

   pip install 'neuralset[tutorials]' neuralfetch

Heavier dependencies (e.g. ``transformers`` for text/image/etc feature
extraction) live in ``[all]`` — see :doc:`Installation <install>` for
the full breakdown.

----

Quickstart
----------

From study to PyTorch batch — pick an example to see the code.

.. raw:: html

   <div class="code-selector" data-quickstart="true">

     <div class="selector-compact">
       <label class="selector-item">
         <span class="selector-label">📋 Example</span>
         <select id="sel-preset">
           <option value="bel-language" selected>🗣️ Language + MEG — Bel2026PetitListenSample</option>
           <option value="li2022-language">🗣️ Language + fMRI — Li2022PetitSample</option>
           <option value="grootswagers-image">🖼️ Image + EEG — Grootswagers2022HumanSample</option>
           <option value="allen-image">🖼️ Image + fMRI — Allen2022MassiveSample</option>
           <option value="fake-classif">🏷️ Classification + fMRI — Fake2025Fmri (no download)</option>
         </select>
       </label>
     </div>

     <div class="code-block-wrapper">
       <div class="code-block-label">
         <i class="fas fa-database"></i> Load study data, configure extractors & segment
       </div>
       <pre><code id="code-data" class="language-python"></code></pre>
     </div>

   </div>

----

Tutorials
---------

Each tutorial walks through one building block of the pipeline.

.. raw:: html

   <div class="pipeline-vertical">

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-table"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Events</div>
         <div class="pipeline-vcard-desc">The universal data format — every recording, stimulus, and annotation is an event.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-events">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/01_events.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-events">
           <pre><code>from neuralset.events import Event
   evt = Event(type="Word", start=1.0,
               duration=0.3, timeline="sub-01")
   print(evt)              # pydantic model
   print(evt.model_dump()) # dict</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/01_events.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-database"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Studies</div>
         <div class="pipeline-vcard-desc">Download datasets and load them as events DataFrames.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-studies">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/02_studies.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-studies">
           <pre><code>study = ns.Study(name="Fake2025Meg",
                    path=ns.CACHE_FOLDER)
   events = study.run()
   print(f"{len(events)} events, "
         f"{events['subject'].nunique()} subjects")</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/02_studies.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-shuffle"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Transforms</div>
         <div class="pipeline-vcard-desc">Filter, split, and enrich events before extraction.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-transforms">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/03_transforms.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-transforms">
           <pre><code>import neuralset as ns
   transform = ns.events.transforms.AddSentenceToWords()
   events = transform(events)
   print(events[events.type == "Sentence"].head())</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/03_transforms.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-cubes"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Extractors</div>
         <div class="pipeline-vcard-desc">Turn events into tensors — brain signals, text embeddings, images, labels.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-extractors">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/04_extractors.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-extractors">
           <pre><code>meg = ns.extractors.MegExtractor(frequency=100.0)
   sample = meg(events, start=0.0, duration=1.0)
   print(f"MEG shape: {sample.shape}")</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/04_extractors.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-scissors"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Segmenter &amp; Dataset</div>
         <div class="pipeline-vcard-desc">Create time-locked segments and iterate with a PyTorch DataLoader.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-segmenter">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/05_segmenter.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-segmenter">
           <pre><code>segmenter = ns.dataloader.Segmenter(
       start=-0.1, duration=0.5,
       trigger_query='type=="Word"',
       extractors=dict(meg=meg),
       drop_incomplete=True)
   dataset = segmenter.apply(events)
   loader = DataLoader(dataset, batch_size=8,
                       collate_fn=dataset.collate_fn)</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/05_segmenter.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>

     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-link"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Chains</div>
         <div class="pipeline-vcard-desc">Compose Study + Transforms into reproducible, cacheable pipelines.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-chains">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/walkthrough/06_chains.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-chains">
           <pre><code>chain = ns.Chain(steps=[
       ns.Study(name="Fake2025Meg",
                path=ns.CACHE_FOLDER),
       transforms.AddSentenceToWords(),
   ])
   events = chain.run()</code></pre>
           <div class="accordion-footer"><a href="auto_examples/walkthrough/06_chains.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>

   </div>

.. raw:: html

   <div class="page-nav">
     <a href="install.html" class="page-nav-btn page-nav-btn--outline">Installation &rarr;</a>
     <a href="auto_examples/walkthrough/index.html" class="page-nav-btn">Start Tutorials &rarr;</a>
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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: neuralset

   Installation <install>
   Tutorials <auto_examples/walkthrough/index>
   Caching & Cluster Execution <caching_and_cluster>
   Encoding & Decoding <encoding_decoding>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   API <reference/reference>
   Glossary <glossary>
   Contributing <extending/contributing.md>
