.. _neuraltrain_tutorials:

Tutorials: the neuraltrain pipeline
====================================

Each tutorial covers one stage of the training pipeline -- from
wiring data loaders through to running experiment sweeps -- with
code you can run and modify.

.. raw:: html

   <div class="pipeline-vertical">
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-layer-group"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Data</div>
         <div class="pipeline-vcard-desc">Use neuralset studies and a Segmenter to build train/val/test loaders.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-data">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/data/01_data.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-data">
           <pre><code>events = self.study.run()&#10;dataset = self.segmenter.apply(events)&#10;dataset.prepare()&#10;for split in ["train", "val", "test"]:&#10;    ds = dataset.select(&#10;        dataset.triggers["split"] == split)&#10;    loaders[split] = DataLoader(ds, ...)</code></pre>
           <div class="accordion-footer"><a href="auto_examples/data/01_data.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-brain"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Model Config</div>
         <div class="pipeline-vcard-desc">Define serializable model configs and build the PyTorch module when shapes are known.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-model">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/models/02_models.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-model">
           <pre><code>from neuraltrain import models&#10;&#10;model_cfg = models.SimpleConvTimeAgg(&#10;    hidden=32, depth=4,&#10;    merger_config=None)&#10;model = model_cfg.build(&#10;    n_in_channels=208,&#10;    n_outputs=4)</code></pre>
           <div class="accordion-footer"><a href="auto_examples/models/02_models.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-bullseye"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Objective</div>
         <div class="pipeline-vcard-desc">Compose losses, metrics, optimizers, and schedulers as typed config objects.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-objective">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/objectives/03_objectives.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-objective">
           <pre><code>loss = CrossEntropyLoss()&#10;metric = Accuracy(&#10;    log_name="acc",&#10;    kwargs={"task": "multiclass",&#10;            "num_classes": 4})&#10;optim = LightningOptimizer(...)</code></pre>
           <div class="accordion-footer"><a href="auto_examples/objectives/03_objectives.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
     <div class="pipeline-varrow">&#x2193;</div>
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-person-running"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Trainer</div>
         <div class="pipeline-vcard-desc">Wrap the model in a Lightning module with train, validation, and test loops.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-trainer">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/trainer/04_trainer.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-trainer">
           <pre><code>brain_module = BrainModule(&#10;    model=brain_model,&#10;    loss=self.loss.build(),&#10;    optim_config=self.optim,&#10;    metrics={m.log_name: m.build()&#10;             for m in self.metrics})&#10;&#10;exp = Experiment(**default_config)&#10;results = exp.run()</code></pre>
           <div class="accordion-footer"><a href="auto_examples/trainer/04_trainer.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
   </div>

.. raw:: html

   <div class="page-nav">
     <a href="install.html" class="page-nav-btn page-nav-btn--outline">&larr; Installation</a>
     <a href="auto_examples/data/01_data.html" class="page-nav-btn">Start: Data Pipeline &rarr;</a>
   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/data/01_data
   auto_examples/models/02_models
   auto_examples/objectives/03_objectives
   auto_examples/trainer/04_trainer
