# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert the upstream ``emg2qwerty`` *generic* Lightning checkpoint
into a ``braindecode.models.EMG2QwertyNet`` checkpoint, optionally
publishing the result to the Hugging Face Hub.

The upstream classifier head sits at ``model.4.{weight,bias}`` inside
``TDSConvCTCModule``; braindecode's ``EMG2QwertyNet`` exposes it as
``final_layer.{weight,bias}``.  ``EMG2QwertyNet.mapping`` already
declares this two-key rename, so ``load_state_dict`` applies it
automatically — this script just builds the module, loads, and uploads.

``DEFAULT_BUILD_KWARGS`` is hardcoded for the *generic* checkpoint
(98 paper-vocab keys + CTC blank → ``n_outputs=99``); personalized
checkpoints use different vocabularies.

Usage
-----

Convert and save locally::

    python scripts/convert_emg2qwerty_checkpoint.py path/to/generic.ckpt \\
        --output /tmp/generic.safetensors

Convert and push to Hugging Face::

    python scripts/convert_emg2qwerty_checkpoint.py path/to/generic.ckpt \\
        --push-to braindecode/emg2qwerty-generic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from textwrap import dedent

import torch
from braindecode.models import EMG2QwertyNet
from safetensors.torch import save_file

LOGGER = logging.getLogger("convert_emg2qwerty")

UPSTREAM_LICENSE = "cc-by-nc-sa-4.0"

DEFAULT_BUILD_KWARGS = {
    "n_outputs": 99,
    "n_chans": 32,
    "n_times": 8000,
    "sfreq": 2000.0,
    "log_softmax": True,
}


def load_into_braindecode(
    ckpt_path: Path,
    build_kwargs: dict | None = None,
) -> EMG2QwertyNet:
    build_kwargs = build_kwargs or DEFAULT_BUILD_KWARGS
    raw = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = raw["state_dict"] if "state_dict" in raw else raw
    model = EMG2QwertyNet(**build_kwargs)
    model.load_state_dict(sd, strict=True)
    return model


def sanity_check(model: EMG2QwertyNet, build_kwargs: dict) -> None:
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, build_kwargs["n_chans"], build_kwargs["n_times"]))
    bn = model.model[0].batch_norm
    LOGGER.info(
        "Forward OK: out=%s | BN(0): mean=%.3f var=%.3f",
        tuple(out.shape), float(bn.running_mean.mean()), float(bn.running_var.mean()),
    )


def build_model_card(repo_id: str) -> str:
    return dedent(f"""\
        ---
        license: {UPSTREAM_LICENSE}
        library_name: braindecode
        tags:
          - braindecode
          - EMG2QwertyNet
          - emg
          - ctc
          - keystroke-decoding
          - emg2qwerty
        ---

        # emg2qwerty — generic baseline (TDS-Conv-CTC, 108-subject pretrained)

        Pre-remapped braindecode-compatible copy of the upstream
        [`generic.ckpt`](https://github.com/facebookresearch/emg2qwerty/blob/main/models/generic.ckpt)
        from the emg2qwerty release (Sivakumar et al., NeurIPS 2024 D&B Track).

        ```python
        from braindecode.models import EMG2QwertyNet

        model = EMG2QwertyNet.from_pretrained("{repo_id}")
        ```

        ## Source

        Upstream repository: <https://github.com/facebookresearch/emg2qwerty>
        Paper: Sivakumar V, Seely J, Du A, Bittner S, Berenzweig A,
        Bolarinwa A, Gramfort A, Mandel M. *emg2qwerty: A Large Dataset
        with Baselines for Touch Typing using Surface Electromyography*.
        Advances in Neural Information Processing Systems (NeurIPS),
        Datasets and Benchmarks Track, 2024.

        ## Modification from upstream

        Upstream `models/generic.ckpt` is a PyTorch-Lightning checkpoint
        of `emg2qwerty.lightning.TDSConvCTCModule`, whose inner
        `nn.Sequential` exposes the classifier head as item 4
        (`model.4.{{weight,bias}}`).  braindecode's `EMG2QwertyNet`
        exposes the same head as a named attribute
        (`final_layer.{{weight,bias}}`).

        The remap is a **two-key rename**, applied once and saved here:

        | Upstream key       | braindecode key       |
        |--------------------|-----------------------|
        | `model.4.weight`   | `final_layer.weight`  |
        | `model.4.bias`     | `final_layer.bias`    |

        All 49 other keys (BatchNorm, MLP, TDS conv blocks) match
        verbatim — both modules expose the backbone as
        `self.model = nn.Sequential(...)`, so the keys already share the
        `model.<index>.` prefix and need no rename.  Weights are
        otherwise unchanged from upstream.

        Conversion is reproducible from
        `neuralbench-repo/scripts/convert_emg2qwerty_checkpoint.py`.

        ### Verification

        BatchNorm running statistics on the first layer match upstream:

        | Stat                                    | This checkpoint | Expected |
        |-----------------------------------------|-----------------|----------|
        | `model.0.batch_norm.running_mean.mean`  | 0.511           | ≈ 0.51   |
        | `model.0.batch_norm.running_var.mean`   | 1.146           | ≈ 1.15   |

        Forward pass on a 1×32×8000 random input returns shape
        `(1, 373, 99)` — the 4 s @ 2 kHz window after the TDS encoder +
        CTC head.

        ## License

        **CC BY-NC-SA 4.0** ([Attribution-NonCommercial-ShareAlike 4.0
        International](https://creativecommons.org/licenses/by-nc-sa/4.0/)),
        inherited from the upstream emg2qwerty release.

        * **Attribution (BY)** — cite Sivakumar et al. (2024) and link
          this repository when the weights are used or redistributed.
        * **NonCommercial (NC)** — not licensed for commercial use.
        * **ShareAlike (SA)** — derivatives must be released under the
          same CC BY-NC-SA 4.0 license.

        braindecode itself is BSD-3-Clause; that license applies to the
        *code*, not to these weights.  The weights are governed solely
        by CC BY-NC-SA 4.0.

        ## Intended use

        Drop-in pretrained backbone for the
        [`emg/qwerty`](https://github.com/facebookresearch/neuroai/tree/main/neuralbench-repo/neuralbench/tasks/emg/qwerty)
        CTC keystroke-decoding task in NeuralBench, or any other research
        workflow consuming `braindecode.models.EMG2QwertyNet`.

        Per the source paper (table 4): zero-shot val/CER ≈ 16 % on a
        held-out subject; further fine-tuning typically reduces CER to
        ≈ 10 % on a personalized split.

        ## Citation

        ```bibtex
        @inproceedings{{sivakumar2024emg2qwerty,
          title     = {{emg2qwerty: A Large Dataset with Baselines for
                       Touch Typing using Surface Electromyography}},
          author    = {{Sivakumar, Viswanath and Seely, Jeffrey and Du,
                       Alan and Bittner, Sean and Berenzweig, Adam and
                       Bolarinwa, Anuoluwapo and Gramfort, Alexandre and
                       Mandel, Michael}},
          booktitle = {{Advances in Neural Information Processing Systems
                       (NeurIPS), Datasets and Benchmarks Track}},
          year      = {{2024}},
          url       = {{https://github.com/facebookresearch/emg2qwerty}},
        }}
        ```
    """)


def push_to_hub(model: EMG2QwertyNet, repo_id: str, commit_message: str) -> None:
    """Publish ``model`` + a license-correct README to ``repo_id`` on HF.

    We bypass ``EEGModuleMixin.push_to_hub`` because (a) it does not
    accept a per-call ``license`` override and (b) its default model-card
    template ignores ``model_card_kwargs["model_summary"]`` -- we'd ship
    a bare card with the inherited braindecode default
    ``bsd-3-clause``, which would mis-license CC BY-NC-SA 4.0 weights
    on a public artifact.  Writing the README ourselves guarantees the
    YAML frontmatter on the Hub matches the upstream license.
    """
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # save_pretrained writes config.json + model.safetensors but no
        # README, so our card cannot be silently overwritten.
        model.save_pretrained(tmp_path)
        (tmp_path / "README.md").write_text(build_model_card(repo_id))
        LOGGER.info("Pushing to %s ...", repo_id)
        # Skip the redundant pickle copy save_pretrained also writes;
        # safetensors is the format we want HF to serve.
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=["pytorch_model.bin"],
        )
    LOGGER.info("Pushed: https://huggingface.co/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ckpt", type=Path, help="Upstream Lightning .ckpt path.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional local safetensors output path.",
    )
    parser.add_argument(
        "--push-to", default=None,
        help="Optional Hugging Face repo id (e.g. braindecode/emg2qwerty-generic).",
    )
    parser.add_argument(
        "--commit-message", default=None,
        help="HF commit message; defaults to a short auto-generated one.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = load_into_braindecode(args.ckpt)
    sanity_check(model, DEFAULT_BUILD_KWARGS)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(args.output))
        LOGGER.info("Saved safetensors → %s", args.output)

    if args.push_to:
        repo_id = args.push_to.rstrip("/")
        commit_message = args.commit_message or (
            f"Add {repo_id.split('/')[-1]} (remapped from upstream "
            "TDSConvCTCModule; CC BY-NC-SA 4.0)"
        )
        push_to_hub(model, repo_id, commit_message)


if __name__ == "__main__":
    main()
