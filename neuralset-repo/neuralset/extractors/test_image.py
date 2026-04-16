# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import pytest
import requests
import torch
from exca import ConfDict

import neuralset as ns
from neuralset.events import etypes


def create_image(
    fp: Path, size: tuple[int, int] = (128, 128), coeff: float = 255
) -> None:
    fp.parent.mkdir(exist_ok=True, parents=True)
    array = np.random.rand(*size, 3) * coeff
    im = PIL.Image.fromarray(array.astype(np.uint8))
    im.save(fp)


def test_image(tmp_path: Path) -> None:
    # A study is just a dataframe of events
    image_fps = [tmp_path / "images" / f"im{k}.jpg" for k in range(2)]
    for fp in image_fps:
        create_image(fp)

    events = pd.DataFrame([dict(start=10.0, filepath=image_fps[0])])
    events["type"] = "Image"
    events["duration"] = 0.5
    events["timeline"] = "foo"

    # For each event, we need to specify how these discrete events
    # can be converted into a dense time series.
    extractor = ns.extractors.HuggingFaceImage(device="cpu")
    data = extractor(events, start=10.0, duration=0.5)
    (n_dims,) = data.shape
    assert n_dims > 0
    assert data.max() > 0
    infra: tp.Any = dict(folder=tmp_path)
    for _ in range(2):
        extractor = ns.extractors.HuggingFaceImage(infra=infra, device="cpu")
        data = extractor(events, start=10.0, duration=0.5)
        (n_dims,) = data.shape
        assert data.max() > 0

        # test prepare
        extractor.prepare(events)
    # check uids
    params = "name=HuggingFaceImage-c5c2ebfa"
    uid = f"neuralset.extractors.image.HuggingFaceImage._get_data,{extractor.infra.version}/{params}"
    assert extractor.infra.uid() == uid
    extractor_keys = set(ConfDict.from_model(extractor, uid=True).keys())
    expected = {
        "allow_missing",
        "name",
        "model_name",
        "event_types",
        "token_aggregation",
        "layer_aggregation",
        "layers",
        "frequency",
        "imsize",
        "aggregation",
        "pretrained",
        "cache_n_layers",
        "infra",  # provides version
    }
    assert extractor_keys == expected
    expected = {
        "name",
        "model_name",
        "imsize",
        "infra",
        "event_types",
        "token_aggregation",
        "layers",
        "layer_aggregation",
        "pretrained",
        "cache_n_layers",
    }
    assert set(extractor.infra.config().keys()) == expected


def test_image_infra_override(tmp_path: Path) -> None:
    infra: tp.Any = {"folder": tmp_path, "cluster": "local"}
    extractor = ns.extractors.HuggingFaceImage(infra=infra)
    assert extractor.infra.gpus_per_node == 1


def make_cat_event(folder: str | Path) -> etypes.Image:
    fp = Path(folder) / "test-data" / "image.jpg"
    if not fp.exists():
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        fp.parent.mkdir(exist_ok=True)
        warnings.warn("Downloading cat image")
        fp.write_bytes(requests.get(url, stream=True, timeout=10).raw.read())
    return etypes.Image(start=0, duration=1, filepath=fp, timeline="blublu")


@pytest.fixture
def cat_event(tmp_path: Path) -> tp.Iterator[etypes.Image]:
    if ns.CACHE_FOLDER.exists():
        tmp_path = ns.CACHE_FOLDER
    yield make_cat_event(tmp_path)


class RecordedOutputs:
    """Creates a callable from another callable with additional extractors:
    - records the output at each call
    - optionally overrides positional arguments
    """

    def __init__(self, func: tp.Callable[..., tp.Any], **overrides: tp.Any) -> None:
        self._func = func
        self._overrides = overrides
        self.outputs: list[tp.Any] = []

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        kwargs.update(self._overrides)
        self.outputs.append(self._func(*args, **kwargs))
        return self.outputs[-1]

    @classmethod
    def as_mocked_method(cls, method: tp.Any, **overrides: tp.Any):
        record = cls(func=method, **overrides)
        assert method.__self__ is not None, "Not a method (not attached to an object)"  # type: ignore
        setattr(method.__self__, method.__name__, record)  # type: ignore
        return record


@pytest.mark.parametrize("token_aggregation", ["mean", "first", None])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_image_token_aggregation(
    cat_event: etypes.Image,
    token_aggregation: tp.Literal["mean", "first", None],
    device: tp.Literal["cuda", "cpu"],
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda not available")
    extractor = ns.extractors.HuggingFaceImage(
        device=device, token_aggregation=token_aggregation
    )
    out = extractor.get_static(cat_event)
    assert out.ndim == 2 if token_aggregation is None else 1


@pytest.mark.parametrize("token_aggregation", ["mean", "first", None])
def test_openai_clip(
    cat_event: etypes.Image,
    token_aggregation: tp.Literal["mean", "first", None],
) -> None:
    extractor = ns.extractors.HuggingFaceImage(
        device="cpu",
        model_name="openai/clip-vit-base-patch32",
        token_aggregation=token_aggregation,
    )
    record = RecordedOutputs.as_mocked_method(
        extractor.model._full_predict, text=["a photo of a cat", "a photo of a dog"]
    )
    latent = next(iter(extractor._get_data([cat_event])))
    if token_aggregation is None:
        assert latent.shape == (50, 768)
    else:
        assert latent.shape == (768,)
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    probs = pred.logits_per_image.softmax(dim=1)
    assert 0.99 < probs[0, 0] < 1


@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("cache_n_layers", (None, 10, 999))
@pytest.mark.parametrize("token_aggregation", ("mean", None))
def test_openai_clip_layer(
    cat_event: etypes.Image,
    pretrained: bool,
    cache_n_layers: int | None,
    token_aggregation: tp.Literal["mean", "first", None],
) -> None:
    extractor = ns.extractors.HuggingFaceImage(
        device="cpu",
        model_name="openai/clip-vit-base-patch32",
        pretrained=pretrained,
        token_aggregation=token_aggregation,
        cache_n_layers=cache_n_layers,
    )
    record = RecordedOutputs.as_mocked_method(
        extractor.model._full_predict, text=["a photo of a cat", "a photo of a dog"]
    )
    latent = next(iter(extractor._get_data([cat_event])))
    expected_shape = {
        (999, None): (13, 50, 768),
        (999, "mean"): (13, 768),
        (10, None): (10, 50, 768),
        (10, "mean"): (10, 768),
        (None, None): (50, 768),
        (None, "mean"): (768,),
    }
    assert latent.shape == expected_shape[(cache_n_layers, token_aggregation)]
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    probs = pred.logits_per_image.softmax(dim=1)
    if pretrained:
        assert 0.99 < probs[0, 0] < 1
    else:
        assert probs[0, 0] < 0.95
    # output
    out = extractor.get_static(cat_event)
    if token_aggregation is None:
        assert out.shape == (50, 768)
    else:
        assert out.shape == (768,)


def test_hf_dinov2(cat_event: etypes.Image) -> None:
    extractor = ns.extractors.HuggingFaceImage(
        device="cpu",
        model_name="facebook/dinov2-small-imagenet1k-1-layer",
        token_aggregation=None,
    )
    latent = next(iter(extractor._get_data([cat_event])))
    assert latent.shape == (257, 384)
    # empty image
    impath = Path(cat_event.filepath).parent / "empty.png"
    create_image(impath, coeff=0)
    empty_im_event = etypes.Image(filepath=impath, timeline="blublu", start=1, duration=1)
    latent2 = next(iter(extractor._get_data([empty_im_event])))
    assert latent2.shape == latent.shape

    # now check labels are correct with the appropriate classif model (hacky)
    extractor = ns.extractors.HuggingFaceImage(  # new cache
        device="cpu",
        model_name="facebook/dinov2-small-imagenet1k-1-layer",
    )
    from transformers import AutoModelForImageClassification

    extractor.model.model = AutoModelForImageClassification.from_pretrained(
        extractor.model_name
    )
    record = RecordedOutputs.as_mocked_method(extractor.model._full_predict)
    try:
        extractor._get_data([cat_event])
    except:  # not the right output layer as we overrode the model
        pass
    assert len(record.outputs) == 1
    pred = record.outputs[0]
    idx = pred.logits.argmax(-1).item()
    assert extractor.model.model.config.id2label[idx] == "tabby, tabby cat"  # type: ignore


@pytest.mark.parametrize("imsize", [None, 512])
def test_hog(cat_event: etypes.Image, imsize: None | int) -> None:
    pytest.importorskip("skimage")
    extractor = ns.extractors.HOG(imsize=imsize)
    features = extractor.get_static(cat_event)
    assert len(features) == (149152 if imsize is None else 127008)
    assert all(features >= 0.0)


@pytest.mark.parametrize("imsize", [None, 512])
def test_lbp(cat_event: etypes.Image, imsize: None | int) -> None:
    pytest.importorskip("skimage")
    pytest.importorskip("cv2")
    extractor = ns.extractors.LBP(imsize=imsize)
    features = extractor.get_static(cat_event)
    assert len(features) == 10
    assert all(features >= 0.0)


@pytest.mark.parametrize("imsize", [None, 512])
def test_color_histogram(cat_event: etypes.Image, imsize: None | int) -> None:
    pytest.importorskip("cv2")
    extractor = ns.extractors.ColorHistogram(imsize=imsize)
    features = extractor.get_static(cat_event)
    assert len(features) == 512
    assert all(features >= 0.0)


def _get_rfft2d_output_dimension(
    return_log_psd: bool,
    return_angle: bool,
    average_channels: bool,
    height: int,
    width: int,
) -> int:
    """Get the output of an RFFT2D latent based on the configuration."""
    k = 1 if (return_log_psd ^ return_angle) else 2
    return (
        (1 if average_channels else 3)  # Number of image channels
        * height  # Number of spectral components in first image dim
        * (width // 2 + 1)  # Number of spectral components in second image dim
        * k  # Account for "viewed-as-float" complex numbers
    )


@pytest.mark.parametrize("n_components_to_keep", [None, 10])
@pytest.mark.parametrize("average_channels", [False, True])
@pytest.mark.parametrize("return_log_psd", [False, True])
@pytest.mark.parametrize("return_angle", [False, True])
@pytest.mark.parametrize("imsize", [None, 512])
def test_rfft2d(
    cat_event: etypes.Image,
    n_components_to_keep: int | None,
    average_channels: bool,
    return_log_psd: bool,
    return_angle: bool,
    imsize: None | int,
) -> None:
    import torchvision.transforms.functional as TF  # noqa

    image = TF.to_tensor(cat_event.read())

    extractor = ns.extractors.RFFT2D(
        n_components_to_keep=n_components_to_keep,
        average_channels=average_channels,
        return_log_psd=return_log_psd,
        return_angle=return_angle,
        imsize=imsize,
    )
    features = extractor.get_static(cat_event)
    assert features.ndim == 1
    width = image.shape[1] if n_components_to_keep is None else n_components_to_keep * 2
    height = image.shape[2] if n_components_to_keep is None else n_components_to_keep * 2
    if imsize is None:
        assert len(features) == _get_rfft2d_output_dimension(
            return_log_psd, return_angle, average_channels, width, height
        )

    # Make sure inverse is same as original image
    if (
        n_components_to_keep is None
        and not average_channels
        and not return_log_psd
        and not return_angle
        and imsize is None
    ):
        image2 = extractor._ifft(features, average_channels, width, height)
        assert torch.allclose(image, image2, atol=1e-6)
