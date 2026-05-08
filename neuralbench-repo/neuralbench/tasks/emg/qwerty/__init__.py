# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC task. Imports below fire ``__init_subclass__`` on
neuralset's pydantic registries (extractor + study + callback configs).
"""

from neuralbench.pl_module import register_ctc_metric

# Side-effect imports: each submodule defines pydantic-discriminated
# subclasses (Study sources, extractors, callback configs) whose
# ``__init_subclass__`` hooks register them with neuralset / neuralbench
# at import time.  Don't drop these "unused" imports — the YAML parser
# can't resolve ``name: Emg2qwerty`` etc. without them.
from . import callbacks, extractors, study  # noqa: F401
from .metrics import CharacterErrorRates


def _ctc_metric_factory_builder(extractor):
    """Return a no-arg metric factory bound to the extractor's vocabulary.

    Captures the extractor's charset at ``Experiment.prepare_pl_module``
    time so multi-experiment grids running different ``vocab_preset``
    values get independent metrics — no shared process-global to clobber.
    """
    cs = getattr(extractor, "charset", None)
    return lambda: CharacterErrorRates(charset_=cs)


register_ctc_metric("qwerty", _ctc_metric_factory_builder)
