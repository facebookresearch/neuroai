# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.

# Ensure real packages take priority over docs/ subdirectory namespace packages.
# The docs/neuralset/, docs/neuralfetch/, etc. directories are RST-only (no
# __init__.py), but Python 3 treats them as namespace packages and they shadow
# the real installed packages when Sphinx runs from docs/.  By inserting the
# source repos at the front of sys.path the PathFinder will find the real
# __init__.py first.
import builtins
import importlib
import os
import re
import sys
import typing
from typing import Dict  # noqa

builtins.Any = typing.Any

_docs_root = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_docs_root)
for _repo in [
    "neuralset-repo",
    "neuraltrain-repo",
    "neuralfetch-repo",
]:
    _p = os.path.join(_repo_root, _repo)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# -- Project information -----------------------------------------------------
project = "neuroai"
copyright = "Meta Platforms, Inc. and affiliates"
author = "FAIR"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
suppress_warnings = [
    "image.not_readable",
    "ref.footnote",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".pytest_cache",
    "sg_execution_times.rst",
    "**/GALLERY_HEADER.rst",
    "neuralset/walkthrough/concepts_overview.rst",
]

# Prefix document path to section labels
autosectionlabel_prefix_document = True
autosummary_generate = True

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Syntax highlighting
pygments_style = "sas"
pygments_dark_style = "monokai"


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/"
html_css_files = [
    "custom.css",
    "conf.css",
    f"{font_awesome}all.min.css",
    f"{font_awesome}fontawesome.min.css",
    f"{font_awesome}solid.min.css",
    f"{font_awesome}brands.min.css",
]

html_js_files = [
    "pipeline-accordion.js",
    "code-selector.js",
    "sidebar-nav.js",
]

html_theme_options = {
    "light_css_variables": {
        "admonition-font-size": "100%",
        "admonition-title-font-size": "100%",
        "color-brand-primary": "#448aff",
        "color-brand-content": "#448aff",
        "color-admonition-title--note": "#448aff",
        "color-admonition-title-background--note": "#448aff10",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448aff",
        "color-brand-content": "#448aff",
        "color-announcement-background": "#935610",
        "color-announcement-text": "#FFFFFF",
    },
    "source_repository": "https://github.com/facebookresearch/neuroai/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/facebookresearch/neuroai",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
}

html_static_path = ["_static"]
html_title = "neuroai"
html_short_title = "neuroai"
html_show_sourcelink = False
html_favicon = "_static/favicon.ico"

# Sphinx copybutton config
copybutton_prompt_text = ">>> "

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "exca": ("https://facebookresearch.github.io/exca/", None),
}

# -- Nitpicky mode: flag unresolved cross-references ------------------------
# Most surviving targets are either (a) signature-parse artefacts where
# Sphinx treats a default/literal value as a class reference, or (b) short
# names (``Tensor``, ``Module``, ``Path``) that lack a qualifying module in
# the source docstring. We suppress those en masse; genuine typos in
# dotted paths should still surface as warnings.
nitpicky = True
nitpick_ignore_regex = [
    # (a) literal values misinterpreted as class refs from signature parsing
    (r"py:class", r"^\w+\s*=.*"),  # keyword=value fragments
    (r"py:class", r"^optional( .*)?$"),
    (r"py:class", r"^[\"'].*"),  # quoted literals
    (r"py:class", r"^\{.*"),  # dict/set literals
    (r"py:class", r"^-?\d[\d._e+-]*$"),  # numeric literals
    (r"py:class", r"^\.\.$"),
    (r"py:class", r"^tp\.Literal$"),
    # (b) malformed type annotations in docstrings (real Python refs never
    # contain whitespace)
    (r"py:class", r".*\s.*"),
    # (c) short names lacking module context
    (r"py:class", r"^(Tensor|Module|Path|PathLike|PIL\.Image)$"),
    # (d) third-party libraries with no intersphinx inventory
    # exca's new `steps` module (Step, Chain, Backend, ...) is not yet
    # published in exca's inventory (WIP upstream).
    (r"py:class", r"^exca\.(Step|Chain|steps\..*)$"),
    (r"py:class", r"^annotated_types\..*"),
    (r"py:(class|func)", r"^huggingface_hub\..*"),
    (r"py:class", r"^_?PydanticUndefined$"),
    (r"py:class", r"^_PydanticGeneralMetadata$"),
    # (e) private helpers referenced in public docstrings
    (r"py:class", r".*\._[A-Za-z]\w*$"),
]


autosummary_generate = True

# -- Shorten verbose default values in signatures ---------------------------


def _shorten_signature(app, what, name, obj, options, signature, return_annotation):
    if signature:
        signature = re.sub(r"MapInfra\([^)]*\)", "MapInfra()", signature)
    return signature, return_annotation


def _resolve_short_paths(app, env, node, contnode):
    """Retry unresolved py-refs using the target's canonical ``__module__``.

    Docstrings routinely use short public paths (e.g. ``neuralset.events.Event``)
    but autodoc registers symbols at their source module
    (``neuralset.events.etypes.Event``), so such refs fail. This handler
    imports the short path, discovers the canonical location, and retries
    once. Genuine typos still fail.
    """
    if node.get("refdomain") != "py":
        return None
    parts = node.get("reftarget", "").split(".")
    # Try the longest importable module prefix, then descend attributes.
    # e.g. ``neuralset.events.Event.from_dict`` → import ``neuralset.events``,
    # getattr ``Event`` → class, getattr ``from_dict`` → method.
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        mod = getattr(obj, "__module__", None)
        if not mod or mod == ".".join(parts[:i]):
            return None  # already at canonical module; nothing to retry
        # Methods/functions: ``__qualname__`` carries the class scope
        # (e.g. ``BaseExtractor.prepare``) that ``__module__`` alone lacks.
        qualname = getattr(obj, "__qualname__", ".".join(parts[i:]))
        canonical = f"{mod}.{qualname}"
        if canonical == ".".join(parts):
            return None
        return env.get_domain("py").resolve_xref(
            env,
            node["refdoc"],
            app.builder,
            node["reftype"],
            canonical,
            node,
            contnode,
        )
    return None


def setup(app):
    from sphinx.events import EventListener

    app.connect("autodoc-process-signature", _shorten_signature)
    app.connect("missing-reference", _resolve_short_paths)

    listeners = app.events.listeners.get("autodoc-skip-member", [])
    for i, ev in enumerate(listeners):
        _orig = ev.handler

        def _make_safe(fn):
            def _safe(app_, what, name, obj, skip, options):
                if name.startswith("__pydantic"):
                    return True
                try:
                    return fn(app_, what, name, obj, skip, options)
                except Exception:
                    return None

            return _safe

        listeners[i] = EventListener(ev.id, _make_safe(_orig), ev.priority)


# -- Linkcode configuration --------------------------------------------------
def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    filepath = info["module"].replace(".", "/")
    if filepath.endswith("infra"):
        filename = info["fullname"].split(".", maxsplit=1)[0].lower().replace("infra", "")
        filepath += f"/{filename}"
    return f"https://github.com/facebookresearch/neuroai/blob/main/neuralset-repo/{filepath}.py"


# -- Sphinx-Gallery configuration --------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": [
        # neuralset
        "neuralset/walkthrough",
        "neuralset/extending",
        # neuraltrain
        "neuraltrain/tutorials/01_data",
        "neuraltrain/tutorials/02_models",
        "neuraltrain/tutorials/03_objectives",
        "neuraltrain/tutorials/04_trainer",
        # neuralfetch
        "neuralfetch/tutorials",
    ],
    "gallery_dirs": [
        # neuralset
        "neuralset/auto_examples/walkthrough",
        "neuralset/auto_examples/extending",
        # neuraltrain
        "neuraltrain/auto_examples/data",
        "neuraltrain/auto_examples/models",
        "neuraltrain/auto_examples/objectives",
        "neuraltrain/auto_examples/trainer",
        # neuralfetch
        "neuralfetch/auto_examples",
    ],
    "filename_pattern": r"/.+\.py$",
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": (
        "neuralset",
        "neuralfetch",
        "neuraltrain",
    ),
    "inspect_global_variables": True,
    "reference_url": {
        "neuralset": None,
        "neuralfetch": None,
        "neuraltrain": None,
    },
    "image_scrapers": ("matplotlib",),
    "matplotlib_animations": True,
    "plot_gallery": True,
    "download_all_examples": True,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}
