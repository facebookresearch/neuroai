"""
Anatomy of a Study
==================

Understand what a Study contains: timelines, events, per-type columns,
and how to compose studies with transforms into a reusable pipeline.

Continue from: :doc:`/neuralfetch/auto_examples/01_plot_fetch_first_study`
"""

# %%
# Setup
# -----

import neuralset as ns

data_folder = ns.CACHE_FOLDER

# %%
# Timelines
# ----------
# A Study yields timelines — each represents one recording session.

study = ns.Study(name="Fake2025Meg", path=data_folder, infra_timelines={"cluster": None})

for tl in study.iter_timelines():
    print(tl)  # dict: {subject: "sample0"}, {subject: "sample1"}

# %%
# Events DataFrame
# -----------------
# ``study.run()`` returns a DataFrame with all events across timelines.

events = study.run()
print(f"Columns: {events.columns.tolist()}")
print(f"Event types: {events['type'].unique().tolist()}")
print(f"Timelines: {events['timeline'].nunique()}")

# %%
# Inspect event types
# --------------------

for t in ["Meg", "Word", "Image", "Audio"]:
    sub = events[events["type"] == t]
    print(f"\n--- {t} ({len(sub)} events) ---")
    cols = [c for c in ["start", "duration", "text", "filepath"] if c in sub.columns]
    print(sub[cols].head(3).to_string())

# %%
# Compose with transforms
# -------------------------
# Studies are steps -- combine with transforms via Chain.
# A list of step dicts is auto-converted to a Chain.

import pydantic


class Experiment(pydantic.BaseModel):
    study: ns.Step


study_config = {
    "name": "Fake2025Meg",
    "path": data_folder,
    "infra_timelines": {"cluster": None},
}
exp = Experiment(
    study=[
        study_config,
        {"name": "ChunkEvents", "event_type_to_chunk": "Audio", "max_duration": 5.0},
        {"name": "QueryEvents", "query": "type in ['Audio', 'Word', 'Meg']"},
    ]
)
events = exp.study.run()
print(
    f"\nAfter transforms: {len(events)} events, types: {events['type'].unique().tolist()}"
)

# %%
# Visualize event distribution
# -----------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
events["type"].value_counts().plot.bar(ax=ax, color="steelblue")
ax.set_ylabel("Count")
ax.set_title("Event type distribution after transforms")
plt.tight_layout()
plt.show()

# %%
# Next steps
# -----------
#
# - :doc:`/neuralfetch/auto_examples/03_create_new_study` — wrap your
#   own dataset as a Study subclass.
# - Use the events DataFrame with ``neuralset`` to build a PyTorch training
#   pipeline (see the neuralset docs).
