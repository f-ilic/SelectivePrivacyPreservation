from torchvision.io import write_png
import torch
import matplotlib.pyplot as plt

from email import parser

import torch
from torch.utils.data import DataLoader


from dataset.db_factory import DBfactory
from utils.info_print import *
from config import parser


from utils.model_utils import (
    set_seed,
)


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--masked", default=None)  # person / background
parser.add_argument("--downsample", type=int, default=None)
parser.add_argument("--interpolation", default=None)  # nearest / bilinear / ...
parser.add_argument("--blur", default=None)  # weak / strong / None
parser.add_argument("--afd_combine_level", type=int, default=None)
parser.add_argument("-combine_masked", action="store_true")
parser.add_argument("--downsample_masked", type=int, default=None)
parser.add_argument("--interpolation_masked", default=None)  # nearest / bilinear / ...
args = parser.parse_args()


def main():
    cfg = args.__dict__.copy()
    # seeds = [1, 90, 986576]
    seeds = [1123, 4, 986576]
    dir_locs = ["1", "2", "3"]
    zipped = list(zip(seeds, dir_locs))

    id_sample = 1  # , 1, 2 <------- Only touch this one
    seed = zipped[id_sample][0]
    dir_loc = zipped[id_sample][1]

    torch.backends.cudnn.benchmark = True
    cfg["datasetname"] = "ipn"
    cfg["privacy"] = True
    cfg["num_workers"] = 0
    cfg["batch_size"] = 1
    cfg["architecture"] = "resnet50"
    cfg["blur"] = None
    cfg["selectively_mask"] = False

    datasetname = cfg["datasetname"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    images = []

    # ----------- Original -----------
    set_seed(seed)
    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    for i, (inputs, masks, flows, labels) in enumerate(test_dataloader):
        images.append(inputs[0])
        break

    # ----------- DOWNSAMPLE -----------

    for downsample in [4, 16]:
        set_seed(seed)
        cfg["downsample"] = downsample
        cfg["interpolation"] = "nearest"

        test_dataset = DBfactory(datasetname, set_split="test", config=cfg)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
        )

        for i, (inputs, masks, flows, labels) in enumerate(test_dataloader):
            images.append(inputs[0])
            break

    # ----------- BLUR -----------

    for blur in ["weak", "strong"]:
        set_seed(seed)
        cfg["downsample"] = None
        cfg["blur"] = blur

        test_dataset = DBfactory(datasetname, set_split="test", config=cfg)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
        )

        for i, (inputs, masks, flows, labels) in enumerate(test_dataloader):
            images.append(inputs[0])
            break

    # ----------- MASKED -----------
    set_seed(seed)
    cfg["downsample"] = None
    cfg["blur"] = None
    cfg["masked"] = "person"
    cfg["mean_fill"] = True

    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    for i, (inputs, masks, flows, labels) in enumerate(test_dataloader):
        images.append(inputs[0])
        break

    t_inv = test_dataset.inverse_normalise
    for en, img in enumerate(images):
        img = (t_inv(img.unsqueeze(1)) * 255).squeeze().byte()
        write_png(img, f"out/{dir_loc}/{en:03d}.png")
        print(f'Wrote to "out/{dir_loc}/{en:03d}.png"')


if __name__ == "__main__":
    main()
