NeuralTrain
===========

NeuralTrain allows training PyTorch models on NeuralSet datasets at scale.

Quick install
-------------

.. code-block:: bash

   pip install neuraltrain

For the full model zoo, Lightning support, and dev tools:

.. code-block:: bash

   pip install 'neuraltrain[dev,lightning,models]'

----

Quick start
-----------

Define training pieces as config objects, then build concrete PyTorch modules
at runtime.

.. code-block:: python

   import torch

   from neuraltrain.losses import base as losses
   from neuraltrain.metrics import base as metrics
   from neuraltrain import models
   from neuraltrain.optimizers import base as optimizers

   # Model
   model_cfg = models.SimpleConvTimeAgg(hidden=32, depth=4, merger_config=None)
   model = model_cfg.build(n_in_channels=208, n_outputs=4)

   # Loss & metrics
   loss_cfg = losses.CrossEntropyLoss()
   metric_cfg = metrics.Accuracy(
       log_name="acc",
       kwargs={"task": "multiclass", "num_classes": 4},
   )

   # Optimizer
   optim_cfg = optimizers.LightningOptimizer(
       optimizer=optimizers.Adam(lr=1e-4),
       scheduler=optimizers.OneCycleLR(
           kwargs={"max_lr": 3e-3, "pct_start": 0.2},
       ),
   )

   x = torch.randn(8, 208, 120)
   y = torch.randint(0, 4, (8,))

   logits = model(x)
   loss = loss_cfg.build()(logits, y)

   metric = metric_cfg.build()
   metric.update(logits, y)

   optimizer_bundle = optim_cfg.build(model.parameters(), total_steps=100)

   print(logits.shape)
   print(float(loss))
   print(metric.compute())
   print(sorted(optimizer_bundle))

----

Tutorials
---------

Each tutorial covers one stage of the training pipeline.

.. raw:: html

   <div class="pipeline-vertical">
     <div class="pipeline-vcard">
       <div class="pipeline-vcard-icon"><i class="fas fa-layer-group"></i></div>
       <div class="pipeline-vcard-body">
         <div class="pipeline-vcard-title">Data</div>
         <div class="pipeline-vcard-desc">Use NeuralSet studies and a Segmenter to build train/val/test loaders.</div>
         <div class="pipeline-sub-links">
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nt-data">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/data/01_data.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nt-data">
           <pre><code>events = self.study.run()&#10;dataset = self.segmenter.apply(events)&#10;dataset.prepare()&#10;loaders = {split: DataLoader(ds, ...)&#10;           for split, ds in ...}</code></pre>
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
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nt-model">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/models/02_models.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nt-model">
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
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nt-objective">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/objectives/03_objectives.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nt-objective">
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
           <a class="pipeline-sub-link pipeline-accordion-toggle" href="#" data-target="acc-nt-trainer">Snippet</a>
           <a class="pipeline-sub-link" href="auto_examples/trainer/04_trainer.html">Tutorial</a>
         </div>
         <div class="pipeline-accordion-panel" id="acc-nt-trainer">
           <pre><code>brain_module = BrainModule(&#10;    model=brain_model,&#10;    loss=self.loss.build(),&#10;    optim_config=self.optim,&#10;    metrics={m.log_name: m.build()&#10;             for m in self.metrics})</code></pre>
           <div class="accordion-footer"><a href="auto_examples/trainer/04_trainer.html">Open full tutorial &#x2192;</a></div>
         </div>
       </div>
     </div>
   </div>

.. raw:: html

   <div class="page-nav">
     <a href="install.html" class="page-nav-btn page-nav-btn--outline">Installation &rarr;</a>
     <a href="auto_examples/data/01_data.html" class="page-nav-btn">Start: Data Pipeline &rarr;</a>
   </div>

----

Citation
--------

If you use ``NeuralTrain`` in your research, please cite
`A foundation model of vision, audition, and language for in-silico
neuroscience <https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/>`_:

.. code-block:: text

   @article{dAscoli2026TribeV2,
     title={A foundation model of vision, audition, and language for in-silico neuroscience},
     author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
     year={2026}
   }

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: NeuralTrain

   Installation <install>
   Tutorials <tutorials>
   Project Example <project_example>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   API <reference/reference>
