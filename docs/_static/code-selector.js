/* code-selector.js — Interactive code builder for encoding/decoding page.
   Assembles two code blocks (data-loading + sklearn) from the current
   dropdown selection (task × direction × device).                         */

document.addEventListener("DOMContentLoaded", function () {

  var dataEl = document.getElementById("code-data");
  var sklearnEl = document.getElementById("code-sklearn");
  if (!dataEl) {
    console.warn("Code selector: Required elements not found");
    return;
  }

  console.log("Code selector: Initialized successfully");

  // ── Copy button functionality ─────────────────────────────────────────
  function addCopyButton(element, content) {
    var wrapper = element.parentElement;
    var existingButton = wrapper.querySelector('.copy-btn');
    if (existingButton) {
      existingButton.remove();
    }

    var copyButton = document.createElement('button');
    copyButton.className = 'copy-btn';
    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
    copyButton.title = 'Copy code';

    copyButton.addEventListener('click', function() {
      // Create a temporary textarea with plain text (no HTML)
      var textarea = document.createElement('textarea');
      textarea.value = content;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);

      // Visual feedback
      var originalHTML = copyButton.innerHTML;
      copyButton.innerHTML = 'Copied ⚡🧠';
      copyButton.classList.add('copied');
      setTimeout(function() {
        copyButton.innerHTML = originalHTML;
        copyButton.classList.remove('copied');
      }, 1500);
    });

    wrapper.appendChild(copyButton);
  }

  // ── Safe Python syntax highlighting (no HTML corruption) ─────────────────
  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function wrapToken(cls, text) {
    return '<span class="' + cls + '">' + escapeHtml(text) + '</span>';
  }

  function isWordChar(ch) {
    return /[A-Za-z0-9_]/.test(ch);
  }

  function isDigit(ch) {
    return /[0-9]/.test(ch);
  }

  function isNameStart(ch) {
    return /[A-Za-z_]/.test(ch);
  }

  var PY_KEYWORDS = {
    "import": true, "from": true, "as": true, "def": true, "class": true,
    "if": true, "elif": true, "else": true, "for": true, "in": true,
    "while": true, "try": true, "except": true, "finally": true, "with": true,
    "return": true, "yield": true, "lambda": true, "and": true, "or": true,
    "not": true, "is": true, "None": true, "True": true, "False": true,
    "break": true, "continue": true, "pass": true, "global": true, "nonlocal": true
  };

  var PY_BUILTINS = {
    "print": true, "len": true, "range": true, "enumerate": true, "zip": true,
    "map": true, "filter": true, "sum": true, "max": true, "min": true,
    "abs": true, "round": true, "type": true, "isinstance": true, "hasattr": true,
    "getattr": true, "setattr": true, "dict": true, "list": true, "tuple": true,
    "set": true, "str": true, "int": true, "float": true, "bool": true
  };

  function highlightPython(code) {
    var i = 0;
    var n = code.length;
    var out = "";

    while (i < n) {
      var ch = code[i];

      if (ch === "#") {
        var cStart = i;
        while (i < n && code[i] !== "\n") i += 1;
        out += wrapToken("highlight-comment", code.slice(cStart, i));
        continue;
      }

      if (ch === '"' || ch === "'") {
        var quote = ch;
        var sStart = i;
        i += 1;
        while (i < n) {
          var cur = code[i];
          if (cur === "\\") {
            i += 2;
            continue;
          }
          if (cur === quote) {
            i += 1;
            break;
          }
          i += 1;
        }
        out += wrapToken("highlight-string", code.slice(sStart, i));
        continue;
      }

      if ((ch === "f" || ch === "b" || ch === "r" || ch === "u" || ch === "F" || ch === "B" || ch === "R" || ch === "U") && i + 1 < n && (code[i + 1] === '"' || code[i + 1] === "'")) {
        var pStart = i;
        var pQuote = code[i + 1];
        i += 2;
        while (i < n) {
          var pCur = code[i];
          if (pCur === "\\") {
            i += 2;
            continue;
          }
          if (pCur === pQuote) {
            i += 1;
            break;
          }
          i += 1;
        }
        out += wrapToken("highlight-string", code.slice(pStart, i));
        continue;
      }

      if (isNameStart(ch)) {
        var wStart = i;
        i += 1;
        while (i < n && isWordChar(code[i])) i += 1;
        var word = code.slice(wStart, i);

        if (PY_KEYWORDS[word]) {
          out += wrapToken("highlight-keyword", word);
          continue;
        }
        if (PY_BUILTINS[word]) {
          out += wrapToken("highlight-builtin", word);
          continue;
        }

        var j = i;
        while (j < n && /\s/.test(code[j])) j += 1;
        if (j < n && code[j] === "(") {
          out += wrapToken("highlight-function", word);
        } else {
          out += escapeHtml(word);
        }
        continue;
      }

      if (isDigit(ch) || (ch === "." && i + 1 < n && isDigit(code[i + 1]))) {
        var numStart = i;
        i += 1;
        while (i < n && /[0-9.]/.test(code[i])) i += 1;
        out += wrapToken("highlight-number", code.slice(numStart, i));
        continue;
      }

      out += escapeHtml(ch);
      i += 1;
    }

    return out;
  }

  // ── Task configs ──────────────────────────────────────────────────────

  var task = {
    language: {
      installComment: '# Requires: `pip install spacy` (or `pip install "neuralset[all]"`) and `python -m spacy download en_core_web_md`',
      stim:
        'stim = ns.extractors.SpacyEmbedding(\n' +
        '    language="english",\n' +
        '    aggregation="trigger",\n' +
        '    infra=infra,\n' +
        ')',
      eventType: "Word",
      isClassification: false,
    },
    image: {
      installComment: '# Requires: `pip install transformers` (or `pip install "neuralset[all]"`)',
      stim:
        'stim = ns.extractors.HuggingFaceImage(\n' +
        '    model_name="facebook/dinov2-small",\n' +
        '    imsize=518,\n' +
        '    aggregation="trigger",\n' +
        '    infra=infra,\n' +
        ')',
      eventType: "Image",
      isClassification: false,
    },
    video: {
      installComment: '# Requires: `pip install transformers` (or `pip install "neuralset[all]"`)',
      stim:
        'stim = ns.extractors.HuggingFaceVideo(\n' +
        '    frequency=4,\n' +
        '    use_audio=False,\n' +
        '    aggregation="trigger",\n' +
        '    infra=infra,\n' +
        ')',
      eventType: "Video",
      isClassification: false,
    },
    classification: {
      stim:
        'stim = ns.extractors.LabelEncoder(\n' +
        '    event_types="Stimulus",\n' +
        '    event_field="description",\n' +
        '    return_one_hot=True,\n' +
        '    aggregation="first",\n' +
        ')',
      eventType: "Stimulus",
      isClassification: true,
    },
    word_classification: {
      stim:
        'stim = ns.extractors.LabelEncoder(\n' +
        '    event_types="Word",\n' +
        '    event_field="text",\n' +
        '    return_one_hot=True,\n' +
        '    aggregation="first",\n' +
        ')',
      eventType: "Word",
      isClassification: true,
    },
  };

  // ── Device configs ────────────────────────────────────────────────────

  var device = {
    meg: {
      neuro:
        'neuro = ns.extractors.MegExtractor(\n' +
        '    frequency=120.0,\n' +
        '    filter=(0.5, 25.0),\n' +
        '    allow_maxshield=True,\n' +
        '    infra=infra,\n' +
        ')',
      start: "-0.1",
      duration: "0.5",
      timeSelection: "X_neuro = neuro_arr[:, :, 48]  # t = 300 ms at 120 Hz",
    },
    eeg: {
      neuro:
        'neuro = ns.extractors.EegExtractor(\n' +
        '    frequency=120.0,\n' +
        '    filter=(0.1, 75.0),\n' +
        '    infra=infra,\n' +
        ')',
      start: "-0.1",
      duration: "0.5",
      timeSelection: "X_neuro = neuro_arr[:, :, 48]  # t = 300 ms at 120 Hz",
    },
    ieeg: {
      neuro:
        'neuro = ns.extractors.IeegExtractor(\n' +
        '    frequency=100.0,\n' +
        '    filter=(0.05, 20.0),\n' +
        '    reference="bipolar",\n' +
        '    picks=("seeg",),\n' +
        '    drop_bads=True,\n' +
        '    infra=infra,\n' +
        ')',
      start: "-0.1",
      duration: "0.5",
      timeSelection: "X_neuro = neuro_arr[:, :, 40]  # t = 300 ms at 100 Hz",
    },
    emg: {
      neuro:
        'neuro = ns.extractors.EmgExtractor(\n' +
        '    frequency=256.0,\n' +
        '    infra=infra,\n' +
        ')',
      start: "-0.1",
      duration: "0.5",
      timeSelection: "X_neuro = neuro_arr[:, :, 102]  # t = 300 ms at 256 Hz",
    },
    fmri: {
      neuro:
        'neuro = ns.extractors.FmriExtractor(\n' +
        '    offset=5,  # 5s hemodynamic delay\n' +
        '    infra=infra,\n' +
        ')',
      start: "0.0",
      duration: "2.0",
      timeSelection: "X_neuro = neuro_arr[:, :, 0]   # single TR",
    },
    fmri_proj: {
      neuro:
        'neuro = ns.extractors.FmriExtractor(\n' +
        '    offset=5,  # 5s hemodynamic delay\n' +
        '    projection={"name": "SurfaceProjector", "mesh": "fsaverage5"},\n' +
        '    infra=infra,\n' +
        ')',
      start: "0.0",
      duration: "2.0",
      timeSelection: "X_neuro = neuro_arr[:, :, 0]   # single TR",
    },
  };

  // ── Study mapping (task × device) ─────────────────────────────────────

  var studyMap = {
    language: { meg: "Gwilliams2022Neural", eeg: "Broderick2018Ephys", fmri: "Nastase2021Narratives", ieeg: "Zada2025Podcast", emg: "Sivakumar2024Emg2qwerty" },
    image:    { meg: "Hebart2023ThingsMeg",   eeg: "Gifford2022Large",   fmri: "Hebart2023ThingsBold", ieeg: "YourStudy", emg: "YourStudy" },
    video:    { meg: "YourStudy",     eeg: "Liu2024Eeg2video",       fmri: "Lahner2024Modeling", ieeg: "YourStudy", emg: "YourStudy" },
    classification: { meg: "Mne2013Sample", eeg: "Cho2017Supporting", fmri: "YourStudy", ieeg: "YourStudy", emg: "Sivakumar2024Emg2qwerty" },
  };

  // Quickstart presets: each maps to a (task, device, study) triple
  var presets = {
    "bel-language":       { taskKey: "language",            deviceKey: "meg",      study: "Bel2026PetitListenSample" },
    "li2022-language":    { taskKey: "language",            deviceKey: "fmri_proj", study: "Li2022PetitSample" },
    "grootswagers-image": { taskKey: "image",               deviceKey: "eeg",      study: "Grootswagers2022HumanSample" },
    "allen-image":        { taskKey: "image",               deviceKey: "fmri",     study: "Allen2022MassiveSample" },
    "fake-classif":       { taskKey: "word_classification", deviceKey: "fmri",     study: "Fake2025Fmri" },
  };

  var isQuickstart = !!document.querySelector('.code-selector[data-quickstart]');

  // ── Template builders ─────────────────────────────────────────────────

  function buildDataBlock(tsk, dev, studyName) {
    var lines = [
      "import neuralset as ns",
      "from torch.utils.data import DataLoader",
      "",
      "# Infra: caching folder (cluster=None runs locally)",
      'infra = {"folder": ns.CACHE_FOLDER / "cache"}',
      "",
      "# 1. Load study"
    ];
    if (studyName === "YourStudy") {
      lines.push(
        "# No built-in dataset for this task/device combo — replace 'YourStudy' below"
      );
    }
    lines.push(
      "study = ns.Study(",
      '    name="' + studyName + '",',
      "    path=ns.CACHE_FOLDER,",
      "    infra=infra,",
      ")",
      "study.download()",
    );
    lines.push(
      "events = study.run()",
      'print(events[["type", "start", "duration", "timeline"]].head(10))',
      "",
      "# 2. Define extractors",
      dev.neuro,
      ""
    );
    if (tsk.installComment) lines.push(tsk.installComment);
    lines.push(tsk.stim);
    lines.push("");
    lines.push("# 3. Segment into a Dataset");
    lines.push("segmenter = ns.dataloader.Segmenter(");
    lines.push("    start=" + dev.start + ",");
    lines.push("    duration=" + dev.duration + ",");
    lines.push('    trigger_query=\'type=="' + tsk.eventType + '"\',');
    lines.push("    extractors=dict(neuro=neuro, stim=stim),");
    lines.push(")");
    lines.push("dset = segmenter.apply(events)  # fast: returns a lazy Dataset");
    lines.push("dset.prepare()  # pre-compute & cache extractor outputs");
    lines.push("");
    lines.push("# 4. Iterate with a DataLoader");
    lines.push("loader = DataLoader(dset, batch_size=32, shuffle=True, collate_fn=dset.collate_fn)");
    lines.push("for batch in loader:");
    lines.push('    print(f"neuro: {batch.data[\'neuro\'].shape}, stim: {batch.data[\'stim\'].shape}")');
    lines.push("    break");
    return lines.join("\n");
  }

  // ── sklearn block ─────────────────────────────────────────────────────

  function buildSklearnBlock(tsk, dev, dir) {
    var isDecoding = (dir === "decoding");

    if (tsk.isClassification && isDecoding) {
      return buildClassificationDecodingBlock(dev);
    }
    if (tsk.isClassification && !isDecoding) {
      return buildClassificationEncodingBlock(dev);
    }
    return buildRegressionBlock(dev, isDecoding);
  }

  function buildRegressionBlock(dev, isDecoding) {
    var comment = isDecoding
      ? "# Decoding: predict stimulus features from brain activity"
      : "# Encoding: predict brain activity from stimulus features";
    var xy = isDecoding
      ? "X, y = X_neuro, stim_arr"
      : "X, y = stim_arr, X_neuro";

    return [
      "import numpy as np",
      "from scipy.stats import pearsonr",
      "from sklearn.linear_model import RidgeCV",
      "from sklearn.preprocessing import StandardScaler",
      "from sklearn.pipeline import make_pipeline",
      "from sklearn.model_selection import KFold",
      "",
      "# Load all data as numpy arrays",
      "batch = dset.load_all()",
      'neuro_arr = batch.data["neuro"].numpy()',
      'stim_arr  = batch.data["stim"].numpy()',
      "",
      "# Select single time point",
      dev.timeSelection,
      "",
      comment,
      xy,
      "",
      "model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 100)))",
      "correlations = []",
      "for train, test in KFold(n_splits=5, shuffle=True, random_state=0).split(X):",
      "    model.fit(X[train], y[train])",
      "    y_pred = model.predict(X[test])",
      "    y_true = y[test]",
      "    r = [pearsonr(t, p)[0] for t, p in zip(y_true.T, y_pred.T)]",
      "    correlations.append(np.mean(r))",
      'print(f"Mean Pearson r = {np.mean(correlations):.3f}")',
    ].join("\n");
  }

  function buildClassificationDecodingBlock(dev) {
    return [
      "import numpy as np",
      "from sklearn.linear_model import RidgeClassifierCV",
      "from sklearn.preprocessing import StandardScaler",
      "from sklearn.pipeline import make_pipeline",
      "from sklearn.model_selection import StratifiedKFold",
      "from sklearn.metrics import balanced_accuracy_score",
      "",
      "# Load all data as numpy arrays",
      "batch = dset.load_all()",
      'neuro_arr = batch.data["neuro"].numpy()',
      'stim_arr  = batch.data["stim"].numpy().ravel()',
      "",
      "# Select single time point",
      dev.timeSelection,
      "",
      "# Decoding: predict stimulus class from brain activity",
      "X, y = X_neuro, stim_arr",
      "",
      "model = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 8, 100)))",
      "scores = []",
      "for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):",
      "    model.fit(X[train], y[train])",
      "    y_pred = model.predict(X[test])",
      "    scores.append(balanced_accuracy_score(y[test], y_pred))",
      'print(f"Mean balanced accuracy = {np.mean(scores):.3f}")',
    ].join("\n");
  }

  function buildClassificationEncodingBlock(dev) {
    return [
      "import numpy as np",
      "from scipy.stats import pearsonr",
      "from sklearn.linear_model import RidgeCV",
      "from sklearn.preprocessing import StandardScaler",
      "from sklearn.pipeline import make_pipeline",
      "from sklearn.model_selection import KFold",
      "",
      "# Load all data as numpy arrays",
      "batch = dset.load_all()",
      'neuro_arr = batch.data["neuro"].numpy()',
      'stim_arr  = batch.data["stim"].numpy().ravel()',
      "",
      "# Select single time point",
      dev.timeSelection,
      "",
      "# Encoding: predict brain activity from stimulus class",
      "X, y = stim_arr.reshape(-1, 1).astype(float), X_neuro",
      "",
      "model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 100)))",
      "correlations = []",
      "for train, test in KFold(n_splits=5, shuffle=True, random_state=0).split(X):",
      "    model.fit(X[train], y[train])",
      "    y_pred = model.predict(X[test])",
      "    y_true = y[test]",
      "    r = [pearsonr(t, p)[0] for t, p in zip(y_true.T, y_pred.T)]",
      "    correlations.append(np.mean(r))",
      'print(f"Mean Pearson r = {np.mean(correlations):.3f}")',
    ].join("\n");
  }

  // ── Rendering ─────────────────────────────────────────────────────────

  function val(id) {
    var el = document.getElementById(id);
    return el ? el.value : null;
  }

  function render() {
    var tskKey, devKey, studyName;

    if (isQuickstart) {
      var p = presets[val("sel-preset")] || presets["bel-language"];
      tskKey = p.taskKey;
      devKey = p.deviceKey;
      studyName = p.study;
    } else {
      tskKey = val("sel-task") || "language";
      devKey = val("sel-device") || "meg";
      studyName = (studyMap[tskKey] || {})[devKey] || "YourStudy";
    }

    var tsk = task[tskKey] || task.language;
    var dev = device[devKey] || device.meg;

    var dataBlock = buildDataBlock(tsk, dev, studyName);
    dataEl.innerHTML = highlightPython(dataBlock);
    addCopyButton(dataEl, dataBlock);

    if (sklearnEl) {
      var dir = val("sel-direction") || "decoding";
      var sklearnBlock = buildSklearnBlock(tsk, dev, dir);
      sklearnEl.innerHTML = highlightPython(sklearnBlock);
      addCopyButton(sklearnEl, sklearnBlock);
    }
  }

  // ── Dropdown handlers ─────────────────────────────────────────────────

  ["sel-task", "sel-direction", "sel-device", "sel-preset"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener("change", render);
  });

  render();
});
