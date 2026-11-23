# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

import neuralset as ns
from neuralfetch._explorer import renderer
from neuralset.events import study


@pytest.fixture
def catalog(monkeypatch):
    info = study.StudyInfo(
        data_shape=(8, 360_000),
        event_types_in_query={"Eeg", "Word"},
        frequency=100,
        num_events_in_query=10,
        num_subjects=3,
        num_timelines=2,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Explorer must not construct or execute study data")

    entries = {
        name: SimpleNamespace(
            __module__=f"{package}.studies.example",
            _info=metadata,
            aliases=('Example "alias"',),
            description="Description <with> & special characters",
            neuro_types=lambda: frozenset({"Eeg"}),
            requirements=("example>=1",),
            run=forbidden,
            download=forbidden,
            url="https://example.org/study?a=1&b=2",
        )
        for name, package, metadata in (
            ("PublicExample", "neuralfetch", info),
            ("ExtensionExample", "extension", info),
            ("NoMetadata", "neuralfetch", None),
        )
    }
    monkeypatch.setattr(ns.Study, "catalog", lambda: entries)
    monkeypatch.setattr(ns.Study, "__new__", forbidden)
    return entries


class _Elements(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.root: dict[str, str | None] = {}
        self.cells: dict[str, dict[str, str | None]] = {}
        self.classes: set[str] = set()
        self.event_attributes: set[str] = set()
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        self.event_attributes.update(name for name, _ in attrs if name.startswith("on"))
        self.classes.update((data.get("class") or "").split())
        if data.get("id") == "study-explorer":
            self.root = data
        if "study-name" in (data.get("class") or "").split():
            self.cells[data["data-name"]] = data


def test_metadata_discovery_and_public_defaults(catalog):
    explorer = renderer.StudyInfoSummaries()
    summaries = explorer.get_summaries()
    assert summaries.name.tolist() == ["ExtensionExample", "PublicExample"]
    assert summaries.n_hours.tolist() == [2.0, 2.0]
    html = explorer.render_html_report()
    elements = _Elements(html)
    assert "<title>Study explorer</title>" in html
    assert 'class="explorer-catalog"' in html
    assert 'class="study-view-nav" aria-label="Study navigation" hidden' in html
    assert "Back to study catalog" in html
    assert "Study not found in this catalog." in html
    assert "explorer-site-nav" not in elements.classes
    assert "study-details-reports" not in elements.classes
    assert "study-details-reports-link" not in elements.classes
    assert elements.cells["PublicExample"]["data-code-url"] == (
        "https://github.com/facebookresearch/neuroai/blob/main/"
        "neuralfetch-repo/neuralfetch/studies/example.py"
    )
    assert elements.cells["ExtensionExample"]["data-code-url"] == ""
    assert json.loads(elements.root["data-device-colors"]) == {
        "Eeg": "#437ED8",
        "Emg": "#B5B5B5",
        "Fmri": "#BA4039",
        "Fnirs": "#E8C13B",
        "Ieeg": "#6D3D92",
        "Meg": "#03B779",
        "Multimodal": "#111827",
    }
    assert 'data-event="extension" data-category="package"' in html
    assert 'data-event="neuralfetch" data-category="package"' in html
    public = renderer.StudyInfoSummaries(
        packages=["neuralfetch", "not_installed"],
        install_commands={
            "neuralfetch": "pip install neuralfetch",
            "extension": "excluded extension command",
        },
    )
    assert public.get_summaries().name.tolist() == ["PublicExample"]
    public_html = public.render_html_fragment()
    assets = files("neuralfetch._explorer").joinpath("static")
    css = assets.joinpath("study_explorer.css").read_text(encoding="utf-8").strip()
    javascript = assets.joinpath("study_explorer.js").read_text(encoding="utf-8").strip()
    packaged_icons = {
        asset.read_text(encoding="utf-8").strip()
        for asset in assets.joinpath("icons").iterdir()
        if asset.name.endswith(".svg")
    }
    assert len(packaged_icons) == 8
    for document in (html, public_html):
        assert css in document and javascript in document
        assert '<script type="module">' in document
        assert 'src="static/study_explorer.js"' not in document
        icons = re.findall(r'<svg class="study-icon"[^>]*>.*?</svg>', document)
        assert len(icons) == 8
        assert set(icons) == packaged_icons
        paths = set()
        for markup in icons:
            icon = ET.fromstring(markup)
            assert icon.attrib["aria-hidden"] == "true"
            assert icon.attrib["focusable"] == "false"
            assert icon.attrib["fill"] == "currentColor"
            assert all(float(size) > 0 for size in icon.attrib["viewBox"].split()[2:])
            assert [child.tag for child in icon] == ["{http://www.w3.org/2000/svg}path"]
            paths.add(icon[0].attrib["d"])
            assert "Font Awesome Free 6.5.2" in markup and "CC BY 4.0" in markup
        assert len(paths) == 8 and all(path.startswith("M") for path in paths)
        assert "<i " not in document and "@font-face" not in document
        assert "cdnjs.cloudflare.com" not in document
    assert set(_Elements(public_html).cells) == {"PublicExample"}
    assert "excluded extension command" not in public_html
    assert json.loads(_Elements(public_html).root["data-install-commands"]) == {
        "neuralfetch": "pip install neuralfetch"
    }
    catalog["ExtensionExample"]._info = None
    assert explorer.get_summaries().name.tolist() == ["PublicExample"]


def test_docs_adapter_preserves_public_catalogue_and_embedding(catalog, tmp_path):
    path = Path(__file__).parents[3] / "docs/scripts/build_study_explorer.py"
    spec = importlib.util.spec_from_file_location("build_study_explorer", path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    target = tmp_path / "_explore_studies.html"
    assert adapter.build_docs_study_explorer(target) == target
    document = target.read_text(encoding="utf-8")
    elements = _Elements(document)
    assert document.startswith('<div id="neuralfetch-study-explorer">')
    assert set(elements.cells) == {"PublicExample"}
    assert elements.cells["PublicExample"]["data-url"] == catalog["PublicExample"].url
    assert elements.cells["PublicExample"]["data-module"] == "neuralfetch.studies.example"
    assert elements.root["data-report-index-url"] == ""
    assert json.loads(elements.root["data-install-commands"]) == {
        "neuralfetch": "pip install neuralfetch"
    }
    assert json.loads(elements.root["data-device-colors"])["Eeg"] == "#2563eb"
    assert "study-details-reports" not in elements.classes
    assert '<script type="module">' in document and "<script src=" not in document


def test_explorer_styles_are_scoped():
    css = re.sub(r"/\*.*?\*/", "", renderer._CSS, flags=re.DOTALL)
    for selector in re.findall(r"([^{}]+)\{", css):
        if selector.strip().startswith("@media"):
            continue
        assert all(
            part.strip().startswith("#study-explorer")
            for part in re.split(r",(?![^()]*\))", selector)
        )
    assert "outline: none" not in css


def test_config_links_escaping_and_trusted_preview(catalog):
    install = 'pip install "extension"\n</script><img src=x onerror=alert(1)>'
    explorer = renderer.StudyInfoSummaries(
        device_colors={"Eeg": "#123456"},
        documentation_url="help.html?a=1&b=2",
        install_commands={"extension": install},
        report_index_url="reports/index.html",
        report_link_label="Browse <reports>",
        source_urls={
            "extension": "https://example.org/project/blob/release/",
            "neuralfetch": "https://github.com/facebookresearch/neuroai/blob/main",
        },
        title='Example </title><script>alert("x")</script>',
    )
    html = explorer.render_html_report()
    elements = _Elements(html)
    assert elements.cells["ExtensionExample"]["data-code-url"] == (
        "https://example.org/project/blob/release/extension-repo/extension/studies/example.py"
    )
    assert elements.root["data-report-index-url"] == "reports/index.html"
    assert json.loads(elements.root["data-install-commands"]) == {"extension": install}
    assert 'href="help.html?a=1&amp;b=2"' in html
    assert 'fill="#123456"' in html and "--nse-eeg: #123456" in html
    assert "Browse &lt;reports&gt; ↗</a>" in html
    assert "<img src=x" not in html and '<script>alert("x")' not in html
    assert "target.dataset.codeUrl" in html
    assert "neuralreport-study-selected" in html
    preview = '<aside id="trusted-preview">Trusted markup</aside>'
    internal = explorer.render_html_report(preview_html=preview)
    assert preview in internal
    assert "study-details-reports-link" not in _Elements(internal).classes
    assert "index.manifest.json" not in html
    with pytest.raises(ValueError, match="report_index_url"):
        renderer.StudyInfoSummaries.model_validate(
            explorer.model_dump()
            | {"report_index_url": "https://example.org:99999/index.html"}
        )
    payload = '</script><img src=x onerror="alert(1)"><svg onload="alert(2)">'
    entry = catalog["PublicExample"]
    entry.description = payload
    entry.aliases = (payload,)
    entry.requirements = (payload,)
    entry.url = "javascript:alert(3)"
    entry.neuro_types = lambda: frozenset({'Eeg" onmouseover="alert(4)'})
    malicious_html = explorer.render_html_report()
    elements = _Elements(malicious_html)
    attributes = elements.cells["PublicExample"]
    assert elements.event_attributes == set()
    assert (
        attributes["data-description"]
        == attributes["data-aliases"]
        == attributes["data-requirements"]
        == payload
    )
    assert attributes["data-url"] == ""
    assert "<img src=x" not in malicious_html and "<svg onload=" not in malicious_html


@pytest.mark.parametrize(
    "config,field",
    [
        ({"report_index_url": "javascript:alert(1)"}, "report_index_url"),
        ({"report_index_url": "//example.org/index.html"}, "report_index_url"),
        ({"report_index_url": "https:///index.html"}, "report_index_url"),
        ({"report_index_url": "index.html\n"}, "report_index_url"),
        ({"documentation_url": "https://user@example.org/help"}, "documentation_url"),
        ({"documentation_url": "help%5c.html"}, "documentation_url"),
        ({"source_urls": {"neuralfetch": "https://example.org/%0arepo"}}, "source_urls"),
        ({"source_urls": {"neuralfetch": "repo/blob/main"}}, "source_urls"),
        ({"source_urls": {"neuralfetch": "https://example.org/repo?"}}, "source_urls"),
        ({"source_urls": {"neuralfetch": "https://example.org/repo#"}}, "source_urls"),
        ({"device_colors": {"Eeg": "red; background:url(x)"}}, "device_colors"),
    ],
)
def test_config_rejects_unsafe_values(config, field):
    with pytest.raises(ValueError, match=field):
        renderer.StudyInfoSummaries.model_validate(config)


def test_empty_metadata_fails_loudly(catalog):
    catalog.clear()
    explorer = renderer.StudyInfoSummaries()
    with pytest.raises(RuntimeError, match="No studies with StudyInfo"):
        explorer.render_html_fragment()
    with pytest.raises(RuntimeError, match="No studies with StudyInfo"):
        explorer.render_html_report()


def test_import_independent_of_optional_packages():
    code = """
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"build_study_explorer", "docs", "neuralbench"}:
            raise ImportError("forbidden dependency: " + fullname)
sys.meta_path.insert(0, Block())
from neuralfetch._explorer import renderer
assert Path(renderer.__file__).parent.resolve() == Path(sys.argv[1]).resolve()
assert renderer.StudyInfoSummaries().report_index_url is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
