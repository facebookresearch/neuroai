"""
.. _tuto_studies:

Studies
=======

A :class:`~neuralset.Study` is an interface to an external dataset: it
knows how to download data, iterate over recording sessions, and load
events. A handful of synthetic/testing studies (``Fake2025Meg``, ``Mne2013Sample``,
…) ship with NeuralSet; the full catalog of public datasets is provided
by the `NeuralFetch <https://github.com/facebookresearch/neuroai>`_
package — no data is bundled, only the code that accesses it.

.. note::

   The examples below use ``Fake2025Meg`` which includes audio events.
   If you haven't already::

      pip install 'neuralset[tutorials]'
"""

# %%
# What is a Study?
# ----------------
#
# A Study encapsulates three responsibilities:
#
# 1. **Download** raw data to a local path (``download()``)
# 2. **Iterate timelines** — each timeline is one recording session
#    for one subject (``iter_timelines()``)
# 3. **Load events** for each timeline into a validated events DataFrame
#    (``run()``)
#
# A timeline groups events that share a common time axis. Timelines
# are useful because they allow (1) loading data from simultaneous
# recordings (e.g. fMRI + EEG) and (2) keeping separate sessions
# (e.g. pre- and post-training).

# %%
# Browsing the Catalog
# --------------------
#
# :meth:`Study.catalog() <neuralset.Study.catalog>` returns
# all registered studies (from ``neuralfetch`` and any other installed
# package):

import neuralset as ns

all_studies = ns.Study.catalog()
print(f"{len(all_studies)} studies available")

# %%
# Loading a Study
# ---------------
#
# Instantiate a :class:`~neuralset.Study` by name.
# ``Study(name="X", ...)`` automatically finds and returns the concrete
# subclass ``X`` — it is equivalent to ``X(...)`` directly.
# Public-data studies are registered through the ``neuralfetch`` package —
# see the :doc:`NeuralFetch documentation </neuralfetch/index>` for the full
# catalog.
#
# Call ``run()`` to load all timelines and concatenate them into a
# single validated events DataFrame. The ``subject``, ``timeline``, and
# ``study`` columns are added automatically.


# ns.CACHE_FOLDER defaults to ~/.cache/neuralset/
study = ns.Study(name="Fake2025Meg", path=ns.CACHE_FOLDER)
events = study.run()

print(f"{len(events)} events across {events.subject.nunique()} subjects")
print(events[["type", "start", "duration", "timeline"]].head(8).to_string())

# %%
# Inspecting Timelines
# ^^^^^^^^^^^^^^^^^^^^
#
# Before loading data, ``study_summary()`` shows one row per timeline
# with metadata columns:

summary = study.study_summary(apply_query=False)
print(summary[["subject", "timeline"]].to_string())

# %%
# Querying Timelines
# ^^^^^^^^^^^^^^^^^^
#
# Use the ``query`` parameter to filter which timelines get loaded.
# The query string is evaluated by :func:`~neuralset.events.transforms.query_with_index`,
# which auto-generates virtual index columns such as ``timeline_index``
# and ``subject_timeline_index`` so you can slice by position:

study_q = ns.Study(
    name="Fake2025Meg",
    path=ns.CACHE_FOLDER,
    query="timeline_index < 1",
)
events_q = study_q.run()
print(f"With query: {events_q.timeline.nunique()} timelines, {len(events_q)} events")

# %%
# Downloading Data
# ----------------
#
# For real studies, call ``study.download()`` before ``run()`` to
# fetch the raw data to the study path. This is a one-time operation:
#
# .. code-block:: python
#
#    study = ns.Study(name="MyStudy2025", path="/data")
#    study.download()        # fetches data to path
#    events = study.run()    # loads and validates events
#
# Many curated studies handle downloading from OpenNeuro automatically.
#
# .. tip::
#
#    ``path`` can be a **shared parent folder** — each study resolves
#    its own subfolder automatically. This lets all studies share one
#    path, so you only need to configure it once.

# %%
# Caching
# -------
#
# Study loading can be cached via the ``infra`` parameter, avoiding
# repeated I/O and validation on subsequent runs:
#
# .. code-block:: python
#
#    study = ns.Study(
#        name="Fake2025Meg",
#        path=ns.CACHE_FOLDER,
#        infra={"backend": "Cached", "folder": "/cache"},
#    )
#    events = study.run()  # first call: loads and caches
#    events = study.run()  # subsequent calls: reads from cache
#
# Cache modes (set via ``infra.mode``):
#
# .. list-table::
#    :header-rows: 1
#
#    * - Mode
#      - Behaviour
#    * - ``"cached"`` (default)
#      - Use cache if available, compute otherwise
#    * - ``"force"``
#      - Recompute this step and all downstream steps
#    * - ``"retry"``
#      - Recompute only if previous run failed
#    * - ``"read-only"``
#      - Only read from cache; error if not cached
#
# See :doc:`/neuralset/caching_and_cluster` for all backend options and
# cluster configuration.
#
# .. tip::
#
#    By default, ``Study`` loads timelines in parallel
#    (``infra_timelines=MapInfra(cluster="processpool")``).
#    This is fast but makes exceptions hard to read (they appear as
#    ``BrokenProcessPool``).  During development or debugging, disable
#    parallelism with ``infra_timelines={"cluster": None}`` to get clear
#    tracebacks.

# %%
# Create Your Own (optional)
# --------------------------
#
# To create a custom study, subclass :class:`~neuralset.Study`
# and implement ``iter_timelines()`` and ``_load_timeline_events()``.
# If your data uses non-standard event types, define them as
# :class:`~neuralset.events.Event` subclasses — they are automatically
# registered and can carry any typed fields you need:

import typing as tp

import pandas as pd

from neuralset.events import etypes


class FaceStimulus(etypes.Event):
    """Custom event for face stimuli."""

    identity: str = ""
    expression: str = "neutral"


class FaceStudy(ns.Study):
    """A minimal study generating synthetic face events."""

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # deactivate multiprocessing (inline classes can't be pickled by workers)
        self.infra_timelines.cluster = None

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subject in ["alice", "bob"]:
            yield dict(subject=subject)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        rows = [
            dict(
                type="FaceStimulus",
                start=float(i),
                duration=2.0,
                identity=f"face_{i:03d}",
                expression=expr,
            )
            for i, expr in enumerate(["happy", "neutral", "sad"])
        ]
        return pd.DataFrame(rows)


# %%
# The same ``run()`` interface works for custom studies:

import tempfile
from pathlib import Path

custom_study = FaceStudy(path=Path(tempfile.mkdtemp()))
custom_events = custom_study.run()

print(custom_events[["type", "start", "identity", "expression", "timeline"]].to_string())

# %%
# For a full worked example with real MEG data, channel positions, and
# NeuralFetch integration, see the
# :doc:`NeuralFetch extending tutorial </neuralfetch/auto_examples/03_create_new_study>`.

# %%
# Next Steps
# ----------
#
# - Modify events with transforms: :doc:`03_transforms`
# - Extract features from events: :doc:`04_extractors`
