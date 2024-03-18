import glob
import os
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
import random
from fractions import Fraction
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
import pandas as pd
import shutil
from os import makedirs
from torch import lerp
from models.valid_models import valid_models
from transforms.transform import DownsampleInterpolateUpsample
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import utils.model_utils as model_utils
import pandas as pd
from os.path import join
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm


class SBU(VisionDataset):
    def __init__(
        self,
        root,
        annot_root,
        set_split,
        transform=None,
        inverse_normalise=None,
        config=None,
    ):
        super(SBU, self).__init__(root)

        self.root = root
        self.cfg = config
        self.transform = transform
        self.inverse_normalise = inverse_normalise

        self.set_split = set_split
        self.annot = pd.read_csv(join(annot_root, f"{set_split}.txt"))
        # self.annot = self.annot[self.annot.set == set_split]

        if self.cfg["privacy"]:
            label = sorted(list(set(list(self.annot.person))))
            self.lookup = "person"
        else:
            label = sorted(list(set(list(self.annot.action))))
            self.lookup = "action"

        self.lbl_to_readable = dict(zip(label, label))
        self.idx_to_class = dict(zip(range(len(label)), label))
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.num_classes = len(self.idx_to_class.keys())
        self.classes = list(self.lbl_to_readable.values())

    def __getitem__(self, idx):
        e = self.annot.iloc[idx]
        video_path = join(self.root, self.annot.iloc[idx]["name"])
        lbl = e[self.lookup]
        label = self.class_to_idx[self.lbl_to_readable[lbl]]
        n_frames = e.t_end - e.t_start

        architecture = valid_models[self.cfg["architecture"]]

        if architecture["num_frames"] == 1:  # this gets the whole clip
            start = int(np.ceil(random.uniform(0, n_frames - 3)))
            end = start + 1
        else:  # this gets the right amount of the clip, based on what num_frames x sample_rate the network takes.
            num_frames = architecture["num_frames"]
            sample_rate = architecture["sample_rate"]
            start = int(
                np.floor(
                    random.uniform(
                        1,
                        n_frames - (num_frames * sample_rate) - 2,
                    ),
                )
            )
            end = start + (num_frames * sample_rate)

        video = self.load_video(video_path, start, end, ext="png")

        dsname = "sbu"
        afd_path = video_path.replace(dsname, f"{dsname}_afd")
        mask_path = video_path.replace(dsname, f"{dsname}_masks")
        flow_path = video_path.replace(dsname, f"{dsname}_flow")

        # assert video.shape[0] == n_frames

        video_data = {}
        video_data["video"] = video

        video_data["mask"] = torch.Tensor([])
        if self.cfg["masked"] != None:
            video_data["mask"] = self.load_video(mask_path, start, end, "jpg")

        video_data["afd"] = torch.Tensor([])
        if self.cfg["afd_combine_level"] != None:
            video_data["afd"] = self.load_video(afd_path, start, end, "png")

        video_data["flow"] = torch.Tensor([])
        if "e2s_" in self.cfg["architecture"]:  # architectures that require flow
            video_data["flow"] = self.load_video(flow_path, start, end, "png")

        video_data["sim"] = torch.Tensor([])
        if self.cfg["selectively_mask"] is True:
            tmp_sims = []
            for target in self.cfg["obfuscate"]:
                sim_path = video_path.replace(dsname, f"sbu_{target}_sim")
                tmp_sims.append(self.load_video(sim_path, start, end, "png"))
            sim = torch.stack(tmp_sims, dim=0).float().mean(dim=0)
            video_data["sim"] = sim
            if self.cfg["iid"] is True:
                video_data["afd"] = torch.randn_like(video_data["sim"]) * 255.0
            else:
                video_data["afd"] = self.load_video(afd_path, start, end, "png")

        video = self.transform(video_data)

        if self.cfg["architecture"] == "slowfast_r50":
            ret_video = {"video": [], "mask": [], "afd": [], "flow": [], "sim": []}
            for i in range(len(video["video"])):
                tmp = model_utils.masking(
                    {
                        "video": video["video"][i],
                        "mask": video["mask"][i] if len(video["mask"]) != 0 else 0,
                        "afd": video["afd"][i] if len(video["afd"]) != 0 else 0,
                        "sim": video["sim"][i] if len(video["sim"]) != 0 else 0,
                    },
                    self.cfg,
                )
                [ret_video[k].append(v) for k, v in tmp.items()]
        else:
            ret_video = model_utils.masking(video, self.cfg)

        if self.cfg["selectively_mask"] is True:
            if self.cfg["architecture"] == "slowfast_r50":
                for i in range(len(ret_video["video"])):
                    ret_video["video"][i] = torch.lerp(
                        ret_video["video"][i],
                        ret_video["afd"][i],
                        ret_video["sim"][i],
                    )
            else:
                ret_video["video"] = torch.lerp(
                    video["video"], video["afd"], video["sim"]
                )

        if self.cfg["privacy"]:
            return (
                ret_video["video"].squeeze(),  # <- has to be either tensor or list.
                ret_video["mask"].squeeze(),
                ret_video["flow"].squeeze(),
                label,
            )
        else:
            return (
                ret_video["video"],
                ret_video["mask"],
                ret_video["flow"],
                label,
            )

    def load_video(self, path, t_start, t_end, ext="jpg"):
        videos = sorted(glob.glob(f"{path}/rgb*.{ext}"))
        vid = []
        for frame_path in videos[t_start - 1 : t_end - 1]:
            frame = read_image(f"{frame_path}")
            vid.append(frame)

        return torch.stack(vid, dim=0).permute(1, 0, 2, 3)

    def __len__(self):
        return len(self.annot)
