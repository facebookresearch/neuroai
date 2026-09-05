API Reference
=============

Download Backends
-----------------

.. currentmodule:: neuralfetch.download

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseDownload
   Dandi
   Eegdash
   Datalad
   Donders
   Dryad
   Figshare
   Huggingface
   Openneuro
   Osf
   Physionet
   Synapse
   Zenodo


Utilities
---------

.. currentmodule:: neuralfetch.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_study_info
   update_source_info
   root_study_folder


Command-line interface
----------------------

The ``neuralfetch`` console script (see :mod:`neuralfetch.commands`) exposes
three subcommands: ``download`` (fetch a registered study's raw dataset),
``study-info`` (compute or update :class:`~neuralset.events.study.StudyInfo`),
and ``export-bids`` (export a study to a BIDS directory tree).

.. currentmodule:: neuralfetch.utils.runner

.. autosummary::
   :toctree: generated/
   :nosignatures:

   download_study
   list_downloadable_studies

.. currentmodule:: neuralfetch.utils.bids

.. autosummary::
   :toctree: generated/
   :nosignatures:

   study_to_bids
   BidsExporter

