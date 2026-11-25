# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types

import numpy as np
import pytest
import torch

from .omni import QwenOmni, _media_tokens, _round, _windows


def test_windows() -> None:
    windows = _windows(37.3, context=30.0, stride=10.0, token_rate=25.0, causal=True)
    assert windows == [(0, 30, 0, 30), (10, 37.3, 30, 37.3)]
    # the last window is shifted back on the grid of gcd(context, stride) = 10 s
    windows = _windows(55.0, context=30.0, stride=20.0, token_rate=13.0, causal=True)
    assert windows == [(0, 30, 0, 30), (20, 50, 30, 50), (30, 55.0, 50, 55.0)]
    # bidirectional windows split their overlaps in the middle, off the 10 s grid
    windows = _windows(55.0, context=30.0, stride=20.0, token_rate=13.0, causal=False)
    assert windows == [(0, 30, 0, 25), (20, 50, 25, 40), (30, 55.0, 40, 55.0)]
    windows = _windows(25.0, context=30.0, stride=10.0, token_rate=25.0, causal=True)
    assert windows == [(0, 25.0, 0, 25.0)]


def test_round() -> None:
    assert [_round(x) for x in (0.5, 1.5, 2.5, 2.4999, -0.5)] == [1, 2, 3, 2, 0]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("duration", [3.0, 14.9, 15.1, 44.9, 100.0])
@pytest.mark.parametrize("context,stride", [(30.0, 10.0), (20.0, 20.0), (30.0, 20.0)])
def test_windows_partition(
    duration: float, context: float, stride: float, causal: bool
) -> None:
    windows = _windows(duration, context, stride, token_rate=25.0, causal=causal)
    assert windows[0][2] == 0 and windows[-1][3] == duration
    for (*_, keep_stop), (_, _, keep_start, _) in zip(windows, windows[1:]):
        assert keep_stop == keep_start
    margin = context - stride if causal else (context - stride) / 2
    for start, stop, keep_start, keep_stop in windows:
        assert start % 10 == 0  # on the grid of gcd(context, stride)
        assert start <= keep_start < keep_stop <= stop <= start + context
        assert keep_start - start >= min(keep_start, margin)
        if causal:
            assert keep_stop == stop
        else:
            assert stop - keep_stop >= min(duration - keep_stop, margin)


def test_media_tokens() -> None:
    # sequence as built by Qwen2_5OmniProcessor with use_audio_in_video=True:
    # text, then 2 s chunks of (2 video steps x 4 spatial tokens, 50 audio tokens)
    text, video, audio = 0, 1, 2
    n_video, n_space, n_audio = 5, 4, 125
    ids, values = [text] * 3, [-1.0] * 3
    for chunk in range(3):
        steps = range(2 * chunk, min(2 * chunk + 2, n_video))
        ids += [video] * len(steps) * n_space
        values += [100 + 10 * t + s for t in steps for s in range(n_space)]
        tokens = range(50 * chunk, min(50 * chunk + 50, n_audio))
        ids += [audio] * len(tokens)
        values += [1000.0 + i for i in tokens]
    ids += [text] * 2
    values += [-1.0] * 2
    layer_offsets = torch.tensor([0.0, 1e4])[:, None, None]
    states = torch.tensor(values)[None, :, None].expand(2, -1, 3) + layer_offsets
    input_ids = torch.tensor(ids)

    out = _media_tokens(states, input_ids, video, n_video)
    assert out.shape == (2, n_video, n_space, 3)
    expected = 100 + 10 * torch.arange(n_video)[:, None] + torch.arange(n_space)
    torch.testing.assert_close(out[..., 0], expected + layer_offsets[:, :, 0, None])

    out = _media_tokens(states, input_ids, audio, n_audio)
    assert out.shape == (2, n_audio, 1, 3)
    expected = (1000 + torch.arange(n_audio))[:, None].float()
    torch.testing.assert_close(out[..., 0], expected + layer_offsets[:, :, 0, None])


@pytest.mark.parametrize("missing", [None, "video", "audio"])
@pytest.mark.parametrize("axis", [-1, -2])  # static (dim,) or timed (dim, time)
def test_select_modality(missing: str | None, axis: int) -> None:
    shape = (3, 4) if axis == -2 else (3,)
    video, audio = np.full(shape, 1.0), np.full(shape, 2.0)
    inputs = {"video": None if missing == "video" else video}
    inputs["audio"] = None if missing == "audio" else audio
    expected = {
        "video": np.zeros(shape) if missing == "video" else video,
        "audio": np.zeros(shape) if missing == "audio" else audio,
    }
    expected["cat"] = np.concatenate([expected["video"], expected["audio"]], axis=axis)
    expected["sum"] = expected["video"] + expected["audio"]
    for modality, out in expected.items():
        # only uses self.modality: no model needed
        extractor = types.SimpleNamespace(modality=modality)
        got = QwenOmni._select_modality(extractor, **inputs, axis=axis)  # type: ignore
        np.testing.assert_array_equal(got, out)
