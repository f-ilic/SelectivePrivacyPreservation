import argparse
import glob
import os
from pathlib import Path

import cv2
import torch
import torchvision
from tqdm import tqdm

import flowiz as fz
from RAFT.core.raft import RAFT
from util import firstframe_warp

# torchvision.set_video_backend("video_reader")

from util import chunker, flow_from_video, set_seed

parser = argparse.ArgumentParser(
    description="Create the motion consistent noise dataset"
)
parser.add_argument("--src", type=str, default="data/ipn")

args = parser.parse_args()


def main(list_of_dirs_to_process, src):
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("RAFT/models/raft-sintel.pth"))
    model = model.module
    model.cuda()
    model.eval()

    for clz in list_of_dirs_to_process:
        print(f"Processing {clz}")
        image_list = sorted(glob.glob(f"{clz}/rgb_*png"))

        fl_img_list = sorted(glob.glob(f"{clz.replace('sbu', 'sbu_flow')}/rgb_*png"))
        afd_image_list = sorted(glob.glob(f"{clz.replace('sbu', 'sbu_afd')}/rgb_*png"))

        if len(image_list) == len(fl_img_list) and len(image_list) == len(
            afd_image_list
        ):
            print(f"already done with {clz}")
            continue

        # process_class(clz)
        chunksize = 301
        for images in chunker(image_list, chunksize):
            set_seed()
            frames = torch.stack(
                [torchvision.io.read_image(img).float() for img in images]
            )
            frame_list = list(frames)  # .permute(0, 3, 1, 2))

            flow = flow_from_video(model, frame_list)

            rgbflows = torch.from_numpy(
                fz.video_flow2color(
                    fz.video_normalized_flow(
                        torch.stack(flow, dim=0),
                    )
                )
            )
            rgbflows = torch.cat([rgbflows, rgbflows[-1, ...].unsqueeze(0)], dim=0)
            afd = firstframe_warp(flow, usecolor=True)

            for im, impath in zip(afd.unbind(0), images):
                save_path = impath.replace("sbu", "sbu_afd")
                Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    save_path,
                    im.numpy(),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

            for im, impath in zip(rgbflows.unbind(0), images):
                save_path = impath.replace("sbu", "sbu_flow")
                Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    save_path,
                    im.numpy(),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the motion consistent noise dataset"
    )
    parser.add_argument("--src", type=str, default="../data/sbu")
    args = parser.parse_args()

    src = args.src
    classes = list(sorted(glob.glob(f"{src}/**/**/**/")))

    processes = []
    for cls in tqdm(classes):
        main([cls], src)
