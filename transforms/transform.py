from tkinter import Pack
import torch
import random
from dataset.db_stats import db_stats
from models.valid_models import valid_models
from torchvision.transforms import (
    Compose,
    Lambda,
    ColorJitter,
    GaussianBlur,
)

from torchvision.transforms.functional import hflip
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    Permute,
    RandomShortSideScale,
)
import pytorchvideo
import pytorchvideo.transforms.functional as F
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.transforms import Resize
from torch.nn import Upsample


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class SelectRandomSingleFrame(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, T, H, W = x.shape
        idx = random.randint(0, T - 1)
        return x[:, idx, ...].unsqueeze(1)


class AddIIDNoise(torch.nn.Module):
    def __init__(self, noise_level):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(x)
        return torch.lerp(x, noise, self.noise_level)


class DownsampleInterpolateUpsample(torch.nn.Module):
    def __init__(self, downsample_factor, interpolate_mode):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.interpolate_mode = interpolate_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, T, H, W = x.shape

        return Compose(
            [
                Resize(H // self.downsample_factor, antialias=True),
                Upsample((H, W), mode=self.interpolate_mode),
            ]
        )(x)


class CustomRandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
    ):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend
        # make deterministic across augmentation of different keys
        self._size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.short_side_scale(
            x, self._size, self._interpolation, self._backend
        )


class CustomRandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
        self.value = random.random()

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if self.value < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def transform_default(cfg, set_split):
    num_frames = valid_models[cfg["architecture"]]["num_frames"]

    datasetname = cfg["datasetname"]
    mean = db_stats[datasetname]["mean"]
    std = db_stats[datasetname]["std"]

    afd_dataset = datasetname.replace(f"{datasetname}", f"{datasetname}_afd")
    afd_mean = db_stats[afd_dataset]["mean"]
    afd_std = db_stats[afd_dataset]["std"]

    flo_dataset = datasetname.replace(f"{datasetname}", f"{datasetname}_flow")
    flow_mean = db_stats[flo_dataset]["mean"]
    flow_std = db_stats[flo_dataset]["std"]

    crop_size = valid_models[cfg["architecture"]]["crop_size"]
    pack = torch.nn.Identity()
    if cfg["architecture"] == "slowfast_r50":
        alpha = valid_models[cfg["architecture"]]["slowfast_alpha"]
        pack = PackPathway(alpha)

    noise = torch.nn.Identity()
    if "noise_level" in cfg and cfg["noise_level"] > 0:
        noise = AddIIDNoise(cfg["noise_level"])

    if cfg["blur"] == "strong":
        blur = GaussianBlur(kernel_size=21, sigma=10.0)
    elif cfg["blur"] == "weak":
        blur = GaussianBlur(kernel_size=13, sigma=10.0)
    else:
        blur = torch.nn.Identity()

    temporal_subsample = UniformTemporalSubsample(num_frames)
    scale = CustomRandomShortSideScale(min_size=crop_size, max_size=320)
    crop = CenterCropVideo(crop_size)

    if datasetname in ["kth", "sbu"]:
        flip = CustomRandomHorizontalFlipVideo(0.5)
    elif datasetname in ["ipn"]:
        flip = torch.nn.Identity()

    downsampleUpsample = torch.nn.Identity()
    if cfg["downsample"] != None:
        downsampleUpsample = DownsampleInterpolateUpsample(
            cfg["downsample"], cfg["interpolation"]
        )

    video_transform = Compose(
        [
            temporal_subsample,
            Lambda(lambda x: x / 255.0),
            blur,
            Permute((1, 0, 2, 3)),
            # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            Permute((1, 0, 2, 3)),
            noise,
            NormalizeVideo(mean, std),
            scale,
            crop,
            flip,
            downsampleUpsample,
            pack,
        ]
    )

    mask_transform = (
        Compose(
            [
                temporal_subsample,
                Lambda(lambda x: x / 255.0),
                scale,
                crop,
                flip,
                pack,
            ]
        )
        if cfg["masked"] != None
        else torch.nn.Identity()
    )

    flow_transform = (
        Compose(
            [
                temporal_subsample,
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(flow_mean, flow_std),
                scale,
                crop,
                flip,
                pack,
            ]
        )
        if "e2s_" in cfg["architecture"]  # architectures that require flow
        else torch.nn.Identity()
    )

    afd_transform = (
        Compose(
            [
                temporal_subsample,
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(afd_mean, afd_std)  # fill with dataset stats
                if cfg["selectively_mask"] is False
                else torch.nn.Identity(),
                scale,
                crop,
                flip,
                pack,
            ]
        )
        if cfg["afd_combine_level"] != None or cfg["selectively_mask"] is True
        else torch.nn.Identity()
    )

    sim_transform = (
        Compose(
            [
                temporal_subsample,
                Lambda(lambda x: x / 255.0),
                scale,
                crop,
                flip,
                pack,
            ]
        )
        if cfg["selectively_mask"] is True
        else torch.nn.Identity()
    )

    return Compose(
        [
            ApplyTransformToKey(key="video", transform=video_transform),
            ApplyTransformToKey(key="mask", transform=mask_transform),
            ApplyTransformToKey(key="flow", transform=flow_transform),
            ApplyTransformToKey(key="afd", transform=afd_transform),
            ApplyTransformToKey(key="sim", transform=sim_transform),
        ]
    )


def InverseNormalizeVideo(cfg):
    mean = db_stats[cfg["datasetname"]]["mean"]
    std = db_stats[cfg["datasetname"]]["std"]
    return NormalizeVideo(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )
