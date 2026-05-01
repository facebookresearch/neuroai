"""
Fetch Your First Study
======================

Download a sample dataset, load it as an events DataFrame, and explore
the event structure — the same workflow that works for all neuralfetch studies.
"""

# %%
# Discover available studies
# --------------------------
#
# ``ns.Study.catalog()`` lists every study registered in neuralfetch,
# including full datasets and their lightweight sample variants.

import collections

import neuralset as ns

all_studies = ns.Study.catalog()
print(f"{len(all_studies)} studies available (full + sample variants)")

modalities = collections.Counter(
    nt for cls in all_studies.values() for nt in cls.neuro_types()
)
print("By modality:", dict(modalities))

# %%
# Inspect a study's metadata
# ---------------------------
#
# Every study exposes class-level metadata — no download needed.

Study = all_studies["Grootswagers2022Human"]
print(f"Description: {Study.description[:100]}...")
print(f"Neuro types: {Study.neuro_types()}")
info = Study._info
print(f"Subjects: {info.num_subjects}, timelines: {info.num_timelines}")

# %%
# Load a sample and preview events
# ---------------------------------
#
# Sample studies are small subsets that download automatically.
# They use the same API as full datasets — only the study name differs.
#
# Here we use ``Fake2025Meg`` which requires no download and
# demonstrates the same events structure you'll see with real data.

study = ns.Study(
    name="Fake2025Meg",
    path=ns.CACHE_FOLDER,
    infra_timelines={"cluster": None},
)
events = study.run()

print(f"Loaded {len(events)} events from {events['subject'].nunique()} subject(s)")
print()
print(events[["type", "start", "duration", "subject"]].head(12).to_string())

# %%
# Explore event types
# --------------------
#
# Every study returns the same schema: a flat DataFrame where brain
# recordings, stimuli, text, and annotations share the same rows,
# distinguished by the ``type`` column.

print("Event types:")
print(events["type"].value_counts().to_string())

# %%
# Each type has its own extra columns:

for t in events["type"].unique():
    sub = events[events["type"] == t]
    extra = [
        c
        for c in ["filepath", "text", "code"]
        if c in sub.columns and sub[c].notna().any()
    ]
    extra_str = extra if extra else "(no extra columns)"
    print(f"  {t}: {extra_str}")

# %%
# Visualise the timeline
# ----------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 3))
types = events["type"].unique()
colors = plt.cm.Set2(range(len(types)))

for i, t in enumerate(types):
    sub = events[events["type"] == t]
    ax.barh(
        i,
        sub["duration"].fillna(0.1),
        left=sub["start"],
        height=0.6,
        label=t,
        color=colors[i],
    )

ax.set_yticks(range(len(types)))
ax.set_yticklabels(types)
ax.set_xlabel("Time (s)")
ax.set_title("Fake2025Meg — event timeline (subject 0)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %%
# Using a real sample dataset
# ----------------------------
#
# Replace ``Fake2025Meg`` with any sample name from the catalog to run
# on real data.  All sample studies download their files automatically:
#
# .. code-block:: python
#
#    # EEG + images (THINGS-EEG2, OpenNeuro subset)
#    study = ns.Study(name="Grootswagers2022HumanSample", path="./data")
#    study.download()
#    events = study.run()
#
#    # MEG + speech (Le Petit Prince, 1 subject)
#    study = ns.Study(name="Bel2026PetitListenSample", path="./data")
#    study.download()
#    events = study.run()
#
# See :doc:`/neuralfetch/samples` for all available sample datasets
# and ready-to-paste loading snippets.
#
# Next: :doc:`/neuralfetch/auto_examples/02_plot_study_anatomy` — timelines,
# event anatomy, and composing studies with transforms.
