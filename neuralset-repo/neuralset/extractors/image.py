# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import pydantic
import torch
from exca import MapInfra
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neuralset.events import etypes
from neuralset.utils import warn_once

from .base import BaseStatic, HuggingFaceMixin

logger = logging.getLogger(__name__)
CLUSTER_DEFAULTS: dict[str, tp.Any] = dict(
    timeout_min=25,
    gpus_per_node=1,
    cpus_per_task=8,
    min_samples_per_job=4096,
)


def _fix_pixel_values(inputs: dict[str, tp.Any]) -> None:
    # prevent nans (happening for uniform images)
    if "pixel_values" in inputs:
        nans = inputs["pixel_values"].isnan()
        if nans.any():
            inputs["pixel_values"][nans] = 0
            inputs["pixel_values"] = inputs["pixel_values"].float()


class _HuggingFace(nn.Module):
    """Wrapper that provides a unified interface for loading and using various HuggingFace
    image models (ViT, DINOv2, CLIP, etc.) with support for hidden state extraction
    from all layers.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "facebook/dinov2-base").
        The model will be loaded from the HuggingFace Hub. Please note that you may have to install additional dependencies to load it correctly.
    output_hidden_states : bool, default=False
        Whether to extract hidden states from all transformer layers. If False, only the hidden state from the
        last layer is returned.
    pretrained : bool, default=True
        Whether to load pretrained weights. If False, initializes the model with
        random weights from the model configuration.
    """

    def __init__(
        self,
        model_name: str,
        output_hidden_states: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        Model: tp.Any  # ignore typing as we'll override the imports
        Processor: tp.Any
        from transformers import AutoModel as Model
        from transformers import AutoProcessor as Processor

        if model_name == "facebook/dpt-dinov2-base-kitti":
            from transformers import DPTForDepthEstimation as Model
        try:
            self.model = Model.from_pretrained(
                model_name, output_hidden_states=output_hidden_states
            )
        except ValueError as e:
            # handle specific cases
            if "VisionEncoderDecoderConfig" in str(e):
                from transformers import VisionEncoderDecoderModel as Model
                from transformers import ViTImageProcessor as Processor
            elif "vit-hybrid" in str(e):
                from transformers import ViTHybridForImageClassification as Model
                from transformers import ViTHybridImageProcessor as Processor
            elif "UperNetConfig" in str(e):
                from transformers import UperNetForSemanticSegmentation as Model
            self.model = Model.from_pretrained(
                model_name, output_hidden_states=output_hidden_states
            )
        if not pretrained:
            self.model = Model.from_config(self.model.config)
        self.model.eval()
        # do_rescale=False because ToTensor does the rescaling
        self.processor = Processor.from_pretrained(model_name, do_rescale=False)
        self.model_name = model_name

    def _full_predict(  # return the raw output, used in tests
        self, images: torch.Tensor, text: str | list[str] = ""
    ) -> tp.Any:
        kwargs: dict[str, tp.Any] = dict(
            images=[i.float() for i in images], return_tensors="pt"
        )
        if text:
            kwargs["text"] = text
        inputs = self.processor(**kwargs)
        _fix_pixel_values(inputs)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            pred = self.model(**inputs)
        return pred

    def forward(self, images) -> torch.Tensor:
        pred = self._full_predict(images)
        pred = getattr(pred, "vision_model_output", pred)  # for clip
        outputs = pred.last_hidden_state
        return outputs


class _ImageDataset(Dataset):
    """PyTorch Dataset for loading and transforming image events.

    This dataset wraps a sequence of image events and applies optional transformations
    to each image when accessed. It is primarily used by the BaseImage class
    to efficiently compute image features in batches.
    """

    def __init__(self, events: tp.Sequence[etypes.Image], transform=None):
        self.events = events
        self.transform = transform

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        try:
            image = self.events[idx].read()
            if self.transform:
                image = self.transform(image)
        except:
            logger.warning("Failed to process image event %s", self.events[idx])
            raise
        return image

    @staticmethod
    def collate_fn(images: list[torch.Tensor]) -> tp.Any:
        # we can't concatenate if the outputs have different sizes
        # for huggingface -> transform is applied later
        if all(i.shape == images[0].shape for i in images):
            return torch.stack(images)
        return images


class BaseImage(BaseStatic, HuggingFaceMixin):
    """Base class for computing features from image events using batch processing.

    This class provides the infrastructure for extracting features from sequences of images
    using neural network models. It handles batching, device management and optional image
    resizing. Subclasses must implement `_extract_batched_latents` to define
    the specific feature extraction logic.

    Parameters
    ----------
    batch_size : int, default=32
        Number of images to process in a batch.
    imsize : int | None, default to None
        Target size for resizing images before processing. If specified, images are
        resized to (imsize, imsize). If None, original image dimensions are preserved.

    """

    # class attributes
    event_types: tp.Literal["Image"] = "Image"
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "transformers>=4.29.2",
        "pillow>=9.2.0",
    )

    # extractor attributes
    batch_size: int = 32
    imsize: int | None = None
    _model: nn.Module = pydantic.PrivateAttr()  # initialized later

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return (
            ["batch_size"]
            + BaseStatic._exclude_from_cls_uid()
            + HuggingFaceMixin._exclude_from_cls_uid()
        )

    def _exclude_from_cache_uid(self) -> list[str]:
        return BaseStatic._exclude_from_cache_uid(
            self
        ) + HuggingFaceMixin._exclude_from_cache_uid(self)

    def _make_transform(self) -> tp.Any:
        from torchvision import transforms

        transfs = [transforms.ToTensor()]
        if self.imsize is not None:
            transfs = [transforms.Resize(self.imsize)] + transfs
        return transforms.Compose(transfs)

    def _get_data(self, events: tp.Sequence[etypes.Image]) -> tp.Iterator[np.ndarray]:
        logger.info(f"Computing {len(events)} image latents")
        dset = _ImageDataset(events, transform=self._make_transform())
        dloader = DataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_ImageDataset.collate_fn,
        )
        if len(events) > 1:
            dloader = tqdm(dloader, desc="Computing image embeddings")  # type: ignore
        # Embed the images in batches
        with torch.no_grad():
            for batch_images in dloader:
                if isinstance(batch_images, torch.Tensor):
                    batch_images = batch_images.to(self.device)
                else:  # should be list of different sizes
                    batch_images = [i.to(self.device) for i in batch_images]
                with torch.no_grad():
                    latents = self._extract_batched_latents(batch_images)
                for latent in latents:
                    # notes: - aggregating with a batch would be slightly more efficient
                    # but code would be messier
                    # - aggregating in cuda avoids transferring too much data to cpu
                    latent = self._aggregate_tokens(latent)
                    yield latent.cpu().numpy()

    def get_static(self, event: etypes.Image) -> torch.Tensor:
        raise NotImplementedError

    def _extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HuggingFaceImage(BaseImage):
    """Compute image embeddings using transformer-based models obtained through HuggingFace API.

    Parameters
    ----------
    model_name : str, default="facebook/dinov2-base"
        HuggingFace model identifier.
    pretrained : bool, default=True
        Whether to load pretrained weights from model. If False, initializes the model with
        random weights from the model configuration.

    """

    # class attributes
    model_name: str = "facebook/dinov2-base"
    # extractor attributes
    pretrained: bool = True
    # for precomputing/caching
    infra: MapInfra = MapInfra(version="v5", **CLUSTER_DEFAULTS)

    def _exclude_from_cache_uid(self) -> list[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["duration", "frequency"]

    @infra.apply(
        item_uid=lambda e: str(e.study_relative_path()),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: tp.Sequence[etypes.Image]) -> tp.Iterator[np.ndarray]:
        for latents in super()._get_data(events):
            if self.cache_n_layers is None:
                latents = self._aggregate_layers(latents)
            yield latents

    def model_post_init(self, log__):
        if self.imsize is not None:
            warn_once(
                f'The effect of "imsize"={self.imsize} might be cancelled by '
                "the HuggingFace processor."
            )
        super().model_post_init(log__)

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model") or self._model is None:
            self._model = _HuggingFace(
                model_name=self.model_name,
                output_hidden_states=True,
                pretrained=self.pretrained,
            )
            self._model.to(self.device)
        return self._model

    def _get_hidden_states(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract hidden_states as n_layers n_layers x (batch, tokens,  features)"""
        # this method is overridden in experimental extractors for more hugging face models
        out = self.model._full_predict(images)  # type: ignore
        out = getattr(out, "vision_model_output", out)  # for clip
        states = out.hidden_states
        if states is None:
            raise RuntimeError(
                f"Model {self.model_name!r} returned hidden_states=None. "
                "This is a known regression in transformers>=5 where some "
                "encoders (CLIP, SAM, ViT) no longer collect intermediate "
                "hidden states. See RELEASE_PLAN.md for the tracking item."
            )
        return states  # type: ignore

    def _extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        states = self._get_hidden_states(images)
        out = torch.cat([x.unsqueeze(1) for x in states], axis=1)  # type: ignore
        # (batch, n_layers, tokens, n_features)
        return out  # type: ignore

    def get_static(self, event: etypes.Image) -> torch.Tensor:
        # layer * patches * size
        latent = next(self._get_data([event]))
        latent = np.array(latent, copy=False)  # make sure it's loaded from memmap
        if self.cache_n_layers is not None:
            latent = self._aggregate_layers(latent)
        # copy needed: memmap arrays are read-only
        return torch.Tensor(np.array(latent, copy=True))


class BaseClassicImageExtractor(BaseStatic):
    """Base class for classic image extractors, e.g. based on numpy, skimage, OpenCV, etc.

    Parameters
    ----------
    imsize : int | None, default to None
        Optionally resize images to imsize before passing them to the model. If None,
        use the original image size.
    """

    # class attributes
    event_types: tp.Literal["Image"] = "Image"
    imsize: int | None = None
    infra: MapInfra = MapInfra(version="v5", **CLUSTER_DEFAULTS)

    @infra.apply(
        item_uid=lambda event: str(event.study_relative_path()),
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[etypes.Image]) -> tp.Iterator[np.ndarray]:
        logger.info("Computing %s for %s images.", type(self).__name__, len(events))

        for event in events:
            image = event.read()
            if self.imsize is not None:
                image = image.resize((self.imsize, self.imsize))

            yield self._get_image_features(np.array(image))

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_static(self, event: etypes.Image) -> torch.Tensor:
        return torch.Tensor(np.asarray(next(self._get_data([event]))))


class RFFT2D(BaseClassicImageExtractor):
    """(Cropped) 2D Fourier spectrum of an image of real values.

    Parameters
    ----------
    n_components_to_keep :
        Number of components of the FFT to keep, starting from low frequencies and
        moving towards higher frequencies. If None, use all components.
    average_channels :
        If True, average RGB channels before taking the FFT (to reduce dimensionality).
    return_log_psd :
        If True, return the flattened log PSD instead of the "viewed-as-real" complex FFT.
    return_angle :
        If True, return the flattened angle. Can be combined with the log PSD.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("torchvision>=0.15.2",)

    n_components_to_keep: int | None = None
    average_channels: bool = True
    return_log_psd: bool = False
    return_angle: bool = False

    _eps: tp.ClassVar[float] = 1e-12

    def _fft(self, image: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft2(image)
        if self.average_channels:
            fft = fft.mean(axis=0, keepdims=True)
        fft = torch.fft.fftshift(fft, dim=1)

        if self.n_components_to_keep is not None:  # Crop FFT by keeping lower frequencies
            mid_point_x = fft.shape[1] // 2
            n = self.n_components_to_keep
            lo = mid_point_x - n
            hi = mid_point_x + n
            fft = fft[:, lo:hi, : n + 1]

        return fft

    @staticmethod
    def _ifft(
        fft: torch.Tensor, average_channels: bool, width: int, height: int
    ) -> torch.Tensor:
        """Convenience function to return in image-space after an FFT.

        Only supports "viewed as real" FFT.
        """
        if fft.ndim == 1:
            fft = fft.reshape(  # Unflatten and convert back to complex
                1 if average_channels else 3,
                width,
                height // 2 + 1,
                2,
            )
            fft = torch.view_as_complex(fft)

        fft = torch.fft.ifftshift(fft, dim=1)
        inv_fft = torch.fft.irfft2(fft).real
        inv_fft = inv_fft / inv_fft.max()
        return inv_fft

    def _get_image_features(self, image: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as TF  # noqa

        fft = self._fft(TF.to_tensor(image))

        out = []
        if self.return_log_psd:
            out.append((fft.abs() ** 2 + self._eps).log())
        if self.return_angle:
            out.append(fft.angle())
        if not (self.return_log_psd or self.return_angle):
            out.append(torch.view_as_real(fft))  # Complex tensor -> Real vector
        features = torch.cat(out, dim=-1).flatten()

        return features


class HOG(BaseClassicImageExtractor):
    """Histogram of oriented gradients (Dalal & Triggs, 2005).

    See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html

    References
    ----------
    .. [1] Dalal, N. and Triggs, B., "Histograms of Oriented Gradients for
           Human Detection," IEEE Computer Society Conference on Computer Vision
           and Pattern Recognition, 2005, San Diego, CA, USA.
           https://ieeexplore.ieee.org/document/1467360
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("scikit-image>=0.22.0",)

    _orientations: tp.ClassVar[int] = 8
    _pixels_per_cell: tp.ClassVar[tuple[int, int]] = (8, 8)
    _cells_per_block: tp.ClassVar[tuple[int, int]] = (2, 2)
    _channel_axis: tp.ClassVar[int] = -1

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        from skimage.feature import hog  # noqa

        features = hog(
            image,
            orientations=self._orientations,
            pixels_per_cell=self._pixels_per_cell,
            cells_per_block=self._cells_per_block,
            channel_axis=self._channel_axis,
            visualize=False,
        )
        return features


class LBP(BaseClassicImageExtractor):
    """Local Binary Pattern (LBP).

    See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    """

    requirements: tp.ClassVar[tuple[str, ...]] = (
        "opencv-python>=4.8.1",
        "scikit-image>=0.22.0",
    )
    _P: tp.ClassVar[int] = 8
    _R: tp.ClassVar[int] = 1
    _method: tp.ClassVar[str] = "uniform"
    _n_bins: tp.ClassVar[int] = 10
    _bin_range: tp.ClassVar[tuple[int, int]] = (0, 10)

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        import cv2  # noqa
        from skimage.feature import local_binary_pattern  # noqa

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # requires grayscale
        lbp = local_binary_pattern(gray, P=self._P, R=self._R, method=self._method)
        hist, _ = np.histogram(lbp.ravel(), bins=self._n_bins, range=self._bin_range)
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        return hist


class ColorHistogram(BaseClassicImageExtractor):
    """Color histogram.
    See https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("opencv-python>=4.8.1",)

    _channels: tp.ClassVar[tuple[int, ...]] = (0, 1, 2)
    _hist_size: tp.ClassVar[tuple[int, ...]] = (8, 8, 8)
    _ranges: tp.ClassVar[tuple[int, ...]] = (0, 256, 0, 256, 0, 256)

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        import cv2  # noqa

        hist = cv2.calcHist([image], self._channels, None, self._hist_size, self._ranges)
        hist = cv2.normalize(hist, hist).flatten()

        return hist
