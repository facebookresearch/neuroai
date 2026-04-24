"""
.. _tuto_events:

Events
======

In neuralset, **everything is an event**: a word displayed on screen,
an MEG recording, an fMRI scan, a button press — each is represented
as an :class:`~neuralset.events.Event` with a position in time.

Events are the common representation that connects data sources
(:term:`studies <Study>`) to the features you extract from them
(:term:`extractors <Extractor>`).  A study produces an events
DataFrame; extractors read it to know *what* to extract and *when*.
"""

# %%
# What is an Event?
# -----------------
#
# An event has three core fields: ``start`` (seconds), ``duration``
# (seconds), and ``timeline`` (which recording session it belongs to).
# Concrete subclasses add modality-specific fields — for example a
# :class:`~neuralset.events.etypes.Word` carries ``text``:

from neuralset.events import etypes

word = etypes.Word(start=1.5, duration=0.3, timeline="sub-01_run-01", text="hello")
print(repr(word))

# %%
# Timelines
# ---------
#
# A :term:`timeline <Timeline>` identifies one continuous time axis —
# typically one recording run for one subject (e.g. ``"sub-01_run-02"``).
# Events that share a timeline share a clock: ``start=1.5`` in one event
# and ``start=2.0`` in another mean they are 0.5 s apart.
#
# A single events DataFrame can hold many timelines (subjects, runs,
# sessions).  Downstream processing (extractors, segmenters) groups by
# timeline automatically, so you rarely need to split manually.

# %%
# The Events DataFrame
# --------------------
#
# The standard way to work with events is a **pandas DataFrame** where
# each row is one event.  This is what :meth:`Study.run()
# <neuralset.Study.run>` produces, but you can also
# build one by hand:

import pandas as pd

from neuralset import events

tl = "sub-01_run-01"
rows = [
    dict(type="Word", start=10.5, duration=0.3, timeline=tl, text="The"),
    dict(type="Word", start=11.0, duration=0.4, timeline=tl, text="cat"),
    dict(type="Word", start=11.6, duration=0.3, timeline=tl, text="sat"),
    dict(type="Sentence", start=10.5, duration=1.4, timeline=tl, text="The cat sat"),
]
df = events.standardize_events(pd.DataFrame(rows))
print(df[["type", "start", "duration", "timeline"]].to_string())

# %%
# :func:`~neuralset.events.standardize_events` validates types, checks
# required fields, and sorts by timeline and start time.
#
# **Filtering by type** is plain pandas:

words = df[df.type == "Word"]
print(f"{len(words)} word events")

# %%
# Event Types
# -----------
#
# neuralset ships with many event subclasses covering neural recordings,
# stimuli, text, and annotations.  The hierarchy lets you filter by a
# parent class to match all its children (e.g. filtering for ``MneRaw``
# matches ``Meg``, ``Eeg``, ``Ieeg``, etc.).
#
# .. code-block:: text
#
#    Event
#    ├── BaseDataEvent                     # events backed by a file
#    │   ├── BaseSplittableEvent           # audio/video/neuro with partial loading
#    │   │   ├── Audio                     # WAV audio files
#    │   │   ├── Video                     # video files
#    │   │   ├── MneRaw                    # brain recordings (MNE format)
#    │   │   │   ├── Meg, Eeg, Emg        # magneto/electro-encephalography, electromyography
#    │   │   │   ├── Ieeg                  # intracranial EEG (sEEG, ECoG)
#    │   │   │   └── Fnirs                 # functional near-infrared spectroscopy
#    │   │   └── Fmri                      # functional MRI (NIfTI)
#    │   ├── Image                         # image files (PIL) with optional caption
#    │   └── Spikes                        # spike-sorted neural data
#    ├── BaseText                          # text-based events
#    │   ├── Text, Sentence, Word          # from paragraph to single word
#    │   ├── Phoneme                       # single phoneme
#    │   └── Keystroke                     # keystroke with text label
#    ├── CategoricalEvent                  # discrete labels / categories
#    │   ├── Action                        # motor execution or imagination
#    │   ├── Stimulus                      # trigger code
#    │   ├── EyeState                      # eyes open / closed
#    │   ├── Artifact, Seizure, ...        # clinical annotations
#    │   └── SleepStage                    # W, N1, N2, N3, R

print("Registered event types:", sorted(etypes.Event._CLASSES.keys()))

# %%
# Creating Events from Dicts
# --------------------------
#
# :meth:`Event.from_dict() <neuralset.events.Event.from_dict>`
# dispatches to the right subclass based on the ``type`` field.
# Fields not in the subclass schema are stored in
# :attr:`~neuralset.events.Event.extra`:

data = {
    "type": "Word",
    "start": 1.0,
    "timeline": "run-01",
    "text": "hello",
    "surprisal": 3.2,
}
event = etypes.Event.from_dict(data)
print(f"class: {type(event).__name__},  extra: {event.extra}")

# %%
# :meth:`~neuralset.events.Event.to_dict` flattens back to a plain dict
# (``extra`` fields become top-level keys):

print(event.to_dict())

# %%
# Next Steps
# ----------
#
# - Learn how studies produce events: :doc:`02_studies`
# - Modify events with transforms: :doc:`03_transforms`
