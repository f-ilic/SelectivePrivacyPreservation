import random
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image
from tqdm import tqdm

import utils.model_utils as model_utils
from models.valid_models import valid_models


class IPN(VisionDataset):
    data_url = "https://github.com/GibranBenitez/IPN-hand"

    def __init__(
        self,
        root,
        annot_root,
        set_split,
        transform=None,
        inverse_normalise=None,
        config=None,
    ):
        super(IPN, self).__init__(root)
        self.root = root
        self.fps = 30
        self.extension = "avi"
        self.cfg = config
        self.transform = transform
        self.inverse_normalise = inverse_normalise

        self.set_split = set_split

        if self.cfg["privacy"]:
            self.annot = pd.read_csv(join(annot_root, f"metadata_{set_split}.csv"))
            self.lbl_to_readable = {"W": "woman", "M": "man"}
            self.idx_to_class = {0: "woman", 1: "man"}
            self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
            self.num_classes = len(self.idx_to_class.keys())
            self.classes = list(self.lbl_to_readable.values())

        else:
            tmp = pd.read_csv(join(annot_root, "classIdx.txt"))
            self.lbl_to_readable = dict(zip(tmp.label, tmp.readable))
            self.idx_to_class = dict(zip(tmp.index, tmp.readable))
            self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
            self.num_classes = len(self.idx_to_class.keys())

            self.annot = pd.read_csv(join(annot_root, f"Annot_{set_split}List.txt"))

            self.classes = list(self.lbl_to_readable.values())

    def __len__(self):
        return len(self.annot)

    def load_frames(self, path, video_name, frame_indices, ext="jpg"):
        vid = []
        for frame_id in frame_indices:
            frame = read_image(f"{path}/{video_name}_{frame_id:06d}.{ext}")
            vid.append(frame)
        return torch.stack(vid, dim=0).permute(1, 0, 2, 3)

    def load_video(self, path, video_name, t_start, t_end, ext="jpg"):
        vid = []
        for frame_id in range(t_start, t_end):
            frame = read_image(f"{path}/{video_name}_{frame_id:06d}.{ext}")
            vid.append(frame)

        return torch.stack(vid, dim=0).permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        e = self.annot.iloc[idx]
        video_path = join(self.root, e.video)
        label = self.class_to_idx[self.lbl_to_readable[e.label]]
        n_frames = e.frames

        architecture = valid_models[self.cfg["architecture"]]

        if architecture["num_frames"] == 1:  # this gets the whole clip
            start = int(np.ceil(random.uniform(0, n_frames - 3)))
            end = start + 1
        else:  # this gets the right amount of the clip, based on what num_frames x sample_rate the network takes.
            start = e.t_start
            num_frames = architecture["num_frames"]
            sample_rate = architecture["sample_rate"]
            min_duration = min(num_frames * sample_rate, n_frames)
            max_duration = n_frames
            end = start + int(
                random.uniform(
                    min_duration,
                    max_duration,
                )
            )  # temporal jitter

        video = self.load_video(video_path, e.video, start, end)

        dsname = "ipn"
        afd_path = video_path.replace(dsname, f"{dsname}_afd")
        mask_path = video_path.replace(dsname, f"{dsname}_masks")
        flow_path = video_path.replace(dsname, f"{dsname}_flow")

        video_data = {}
        video_data["video"] = video

        video_data["mask"] = torch.Tensor([])
        if self.cfg["masked"] != None:
            video_data["mask"] = self.load_video(mask_path, e.video, start, end, "jpg")

        video_data["afd"] = torch.Tensor([])
        if self.cfg["afd_combine_level"] != None:
            video_data["afd"] = self.load_video(afd_path, e.video, start, end, "png")

        video_data["flow"] = torch.Tensor([])
        if "e2s_" in self.cfg["architecture"]:  # architectures that require flow
            video_data["flow"] = self.load_video(flow_path, e.video, start, end, "png")

        video_data["sim"] = torch.Tensor([])
        if self.cfg["selectively_mask"] is True:
            tmp_sims = []
            for target in self.cfg["obfuscate"]:
                sim_path = video_path.replace(dsname, f"ipn_{target}_sim")
                tmp_sims.append(self.load_video(sim_path, e.video, start, end, "jpg"))
            sim = torch.stack(tmp_sims, dim=0).float().mean(dim=0)
            video_data["sim"] = sim
            video_data["afd"] = self.load_video(afd_path, e.video, start, end, "png")

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


if __name__ == "__main__":
    ds = IPN(
        root="data/ipn/",
        annot_root="data/splits/ipn",
        set_split="train",
    )

    dl = DataLoader(
        ds,
        1,
        num_workers=16,
        shuffle=False,
        drop_last=False,
    )

    for sample, label in tqdm(dl):
        # print(f"{sample[0]}, {label}")
        pass
