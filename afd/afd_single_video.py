import argparse
from ast import Slice
import os
from os import listdir, mkdir
from os.path import isfile, join
from pathlib import Path
import flowiz as fz

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_video, write_video

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from SliceViewer import SliceViewer
from util import batch_warp, upsample_flow, warp
from generate_afd import flow_from_video, firstframe_warp

parser = argparse.ArgumentParser()
parser.add_argument(
    "videopath", type=str, help="Path to a video you want to see as AFD"
)
import torchvision

torchvision.set_video_backend("video_reader")
from save_helper import save_tensor_as_img, save_tensor_list_as_gif


def main():
    parser.add_argument(
        "--images", type=str, default=None, help="Path where to save individual images"
    )
    parser.add_argument(
        "--gif", type=str, default=None, help="Path where to save output as gif"
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=1,
        help="Upsample image factor, if image too small to calculate optical flow reliably",
    )

    args = parser.parse_args()
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("RAFT/models/raft-sintel.pth"))
    model = model.module
    model.cuda()
    model.eval()

    video_path = args.videopath

    frames = (read_video(video_path)[0]).float()
    frame_list = list(frames.permute(0, 3, 1, 2))[::2]

    video = torchvision.io.VideoReader(video_path, "video")
    fps = video.get_metadata()["video"]["fps"][0]
    frame_duration_ms = 50

    flow = flow_from_video(model, frame_list, upsample_factor=args.upsample)
    afd = firstframe_warp(
        flow,
        usecolor=False,
        # seed_image="/home/filip/projects/AppearanceFreeActionRecognition/tmp.jpg",
    )
    rgbflow = fz.video_flow2color(fz.video_normalized_flow(torch.stack(flow, dim=0)))

    tmp = torch.cat(
        [
            # frames[:-1, :, 50:-30] / 255.0,
            afd[:-1, :, 50:-30] / 255.0,
            # torch.from_numpy(rgbflow[:, :, 50:-30])
            # / 255.0,
        ],
        dim=2,
    )
    if args.gif != None:
        save_tensor_list_as_gif(
            list(tmp.permute(0, 3, 1, 2)), path=args.gif, duration=frame_duration_ms
        )

    if args.images != None:
        print(
            "to make a high quality gif of the generated images:  gifski --fps 30 -o file.gif --quality 100 *.png"
        )
        Path(join(args.images)).mkdir(parents=True, exist_ok=True)
        for i in range(tmp.shape[0]):
            save_tensor_as_img(
                tmp[i].permute(2, 0, 1), path=f"{args.images}/{i:04d}.png"
            )

    # SliceViewer(tmp, figsize=(15, 5))


if __name__ == "__main__":
    main()
