import argparse
import glob
import os
from os.path import join

import torch
import torch.nn.functional as F
from torchvision.datasets.utils import list_dir
from torchvision.io import write_jpeg
from tqdm import tqdm
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Create masks for IPN dataset")
parser.add_argument("--src", type=str, default="data/ipn")
parser.add_argument("--dst", type=str, default="data/ipn_masks")
args = parser.parse_args()

src = args.src
dst = args.dst
classes = list(sorted(list_dir(src)))


def check(path1, path2, clz):
    if len(glob.glob(path1)) == len(glob.glob(path2)):
        print(f"[  OK  ]")
        return
    else:
        print(f"[ FAIL ]")
        return clz


def check_integrity(src_dir, dst_dir):
    src_images = sorted(glob.glob(f"{src_dir}/*.png")) + sorted(
        glob.glob(f"{src_dir}/*.jpg")
    )
    dst_images = sorted(glob.glob(f"{dst_dir}/*.png")) + sorted(
        glob.glob(f"{dst_dir}/*.jpg")
    )

    assert len(src_images) == len(dst_images)


def process_class(clz):
    model = YOLO("yolo/yolov8n-seg.pt")
    print(f"Processing {clz}")
    os.makedirs(f"{dst}/{clz}", exist_ok=True)

    image_list = sorted(glob.glob(f"{src}/{clz}/*png")) + sorted(
        glob.glob(f"{src}/{clz}/*jpg")
    )
    for image in image_list:
        dst_image = f"{dst}/{clz}/{os.path.basename(image)}"
        results = model(source=image, classes=0, verbose=False, stream=True)

        all_masks = extract_masks_from_results(results)
        write_jpeg(
            (torch.stack(all_masks).repeat(3, 1, 1) * 255).to(torch.uint8),
            dst_image,
        )
    check_integrity(join(src, clz), join(dst, clz))


def extract_masks_from_results(results):
    all_masks = []
    for r in results:
        if r.masks is not None:
            mask = torch.from_numpy(r.masks.cpu().numpy().data).mean(0) > 0
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=r.orig_shape,
                mode="nearest",
            ).squeeze()

        else:
            mask = torch.zeros(r.orig_shape) > 0  # empty mask
        all_masks.append(mask)
    return all_masks


for clz in tqdm(sorted(classes)):
    process_class(clz)
