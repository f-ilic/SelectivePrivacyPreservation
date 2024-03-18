import argparse

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor

# torchvision.set_video_backend("video_reader")
from pathlib import Path

from PIL import Image

from helpers import (
    get_best_matching_descriptor,
    save_similarity_from_descriptor,
    str2bool,
)


def parse():
    parser = argparse.ArgumentParser(
        description="Facilitate similarity inspection between two images."
    )

    parser.add_argument(
        "-descriptorpath",
        type=str,
        default="output/descriptors/test.pt",
        help="The descriptor to compare to",
    )

    parser.add_argument(
        "-imagesdir",
        type=str,
        default="input/",
        help="Directory with images to compare to compare the descriptor to and compute the smiliarity",
    )

    parser.add_argument(
        "--model_type",
        default="dino_vits8",
        type=str,
        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""",
    )
    parser.add_argument(
        "--facet",
        default="key",
        type=str,
        help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""",
    )
    parser.add_argument(
        "--layer",
        default=11,
        type=int,
        help="layer to create descriptors from.",
    )

    parser.add_argument(
        "--use_targeted",
        default="False",
        type=str2bool,
        help="Match descriptor to first image in sequence, or use raw.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    load_size = 224
    bin = False
    stride = 4

    images = []
    for ext in ("*.jpg", "*.png"):
        images.extend(list(Path(args.imagesdir).glob(ext)))

    image_strings = [str(p) for p in sorted(images)]

    frame_list = [
        ToTensor()(
            Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(
                Image.open(i).convert("RGB")
            )
        )
        for i in image_strings
    ]

    videoname = args.imagesdir.split("/")[-2]
    descriptor = torch.load(args.descriptorpath)
    descriptor_name = args.descriptorpath.split("/")[-1].split(".")[0]

    if args.use_targeted:
        descriptor = get_best_matching_descriptor(descriptor, image_strings[0])

    with torch.no_grad():
        save_similarity_from_descriptor(
            descriptor,
            videoname,
            image_strings,
            load_size,
            args.layer,
            args.facet,
            bin,
            stride,
            args.model_type,
            prefix_savedir="output/similarities/",
            name=descriptor_name,
        )
