import argparse
import os
import torch
from torchvision.utils import save_image
import torchvision
from tqdm import tqdm
from glob import glob

# torchvision.set_video_backend("video_reader")
from pathlib import Path

from matcher.helpers import (
    str2bool,
    get_best_matching_descriptor,
    similarity_from_descriptor,
)


def check_integrity(src_dir, dst_dir):
    src_images = sorted(glob(f"{src_dir}/*.png")) + sorted(glob(f"{src_dir}/*.jpg"))
    dst_images = sorted(glob(f"{dst_dir}/*.png")) + sorted(glob(f"{dst_dir}/*.png"))
    return len(src_images) == len(dst_images)


def process_video(args, imagesdir, ds):
    images = []
    for ext in ("*.jpg", "*.png"):
        images.extend(list(Path(imagesdir).glob(ext)))

    image_strings = [str(p) for p in sorted(images)]

    descriptor = torch.load(args.descriptorpath)
    if args.use_targeted:
        targeted_descr = get_best_matching_descriptor(descriptor, image_strings[0])
    else:
        targeted_descr = descriptor
    name = args.descriptorpath.split("/")[-1].split(".")[0]

    ok = False
    if os.path.isdir(os.path.dirname(image_strings[0].replace(ds, f"{ds}_{name}_sim"))):
        ok = check_integrity(
            os.path.dirname(image_strings[0]),
            os.path.dirname(image_strings[0]).replace(ds, f"{ds}_{name}_sim"),
        )

        if ok:
            print(f"[   OK   ]{imagesdir}")
            return

        if not ok:
            print(f"[ FIXING ]{imagesdir}")

    with torch.no_grad():
        images = similarity_from_descriptor(
            targeted_descr,
            image_strings,
            args.load_size,
            args.layer,
            args.facet,
            args.bin,
            args.stride,
        )

        for image, path in zip(images, image_strings):
            out_path = path.replace(ds, f"{ds}_{name}_sim")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image(
                image,
                out_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-descriptorpath",
        type=str,
        default="output/descriptors/eyes.pt",
        help="The descriptor to compare to",
    )

    parser.add_argument(
        "--load_size", default=224, type=int, help="load size of the input image."
    )

    parser.add_argument(
        "--stride",
        default=4,
        type=int,
        help="stride of first convolution layer.small stride -> higher resolution.",
    )

    parser.add_argument(
        "--model_type",
        default="dino_vits8",
        type=str,
        help="type of model to extract. Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |  vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]",
    )
    parser.add_argument(
        "--facet",
        default="key",
        type=str,
        help="""facet to create descriptors from. options: ['key' | 'query' | 'value' | 'token']""",
    )
    parser.add_argument(
        "--layer", default=11, type=int, help="layer to create descriptors from."
    )
    parser.add_argument(
        "--bin",
        default="False",
        type=str2bool,
        help="create a binned descriptor if True.",
    )

    parser.add_argument(
        "--use_targeted",
        default="False",
        type=str2bool,
        help="Match to fist image in sequence if True.",
    )

    parser.add_argument(
        "--dataset",
        default="key",
        type=str,
        help="""options: [ 'ipn' | 'kth' | 'sbu' ]""",
    )

    args = parser.parse_args()
    ds = args.dataset
    root_dir = f"data/{ds}/"

    tmp = os.listdir(root_dir)
    all_images_dirs = [os.path.join(root_dir, x) for x in tmp]
    for imagesdir in tqdm(all_images_dirs, position=0):
        process_video(args, imagesdir, ds)
