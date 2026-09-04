Models
======

This section documents the **foundation models** shipped with NeuralBench and
the quirks (channel-mapping assumptions, pretraining-data overlap, dropout
overrides) that need to be considered when evaluating them.  Task-specific
architectures (``eegnet``, ``deep4net``, ``eegconformer``, ``atcnet``,
``bdtcn``, ``ctnet``, ``shallow_fbcsp_net``, ``simpleconv_time_agg``) are
off-the-shelf braindecode models with no NeuralBench-specific configuration;
see their upstream documentation or ``models/<name>.yaml`` for hyperparameters.

.. toctree::
    :glob:
    :titlesonly:

    *
