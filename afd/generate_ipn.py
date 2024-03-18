import argparse
import glob
import os
from pathlib import Path

import cv2
import torch
import torchvision
from torchvision.datasets.utils import list_dir
from tqdm import tqdm

import flowiz as fz
from RAFT.core.raft import RAFT

# torchvision.set_video_backend("video_reader")
from util import chunker, flow_from_video, set_seed, firstframe_warp

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
    model.load_state_dict(torch.load("afd/RAFT/models/raft-sintel.pth"))
    model = model.module
    model.cuda()
    model.eval()

    for clz in list_of_dirs_to_process:
        print(f"Processing {clz}")
        image_list = sorted(glob.glob(f"{src}/{clz}/*jpg"))
        flow_image_list = sorted(
            glob.glob(f"{src.replace('ipn', 'ipn_flow')}/{clz}/*jpg")
        )
        afd_image_list = sorted(
            glob.glob(f"{src.replace('ipn', 'ipn_afd')}/{clz}/*jpg")
        )

        if len(image_list) == len(flow_image_list) and len(image_list) == len(
            afd_image_list
        ):
            print(f"already done with {clz}")
            continue

        # process_class(clz)
        chunksize = 301
        for images in tqdm(
            chunker(image_list, chunksize),
            total=image_list.__len__() // chunksize,
        ):
            set_seed()
            frames = torch.stack(
                [torchvision.io.read_image(img).float() for img in images]
            )
            frame_list = list(frames)  # .permute(0, 3, 1, 2))

            flow = flow_from_video(model, frame_list, upsample_factor=4)

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
                save_path = impath.replace("ipn", "ipn_afd")
                Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    save_path,
                    im.numpy(),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

            for im, impath in zip(rgbflows.unbind(0), images):
                save_path = impath.replace("ipn", "ipn_flow")
                Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    save_path,
                    im.numpy(),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )


if __name__ == "__main__":
    src = args.src
    classes = list(sorted(list_dir(src)))

    processes = []
    for cls in classes:
        main([cls], src)
