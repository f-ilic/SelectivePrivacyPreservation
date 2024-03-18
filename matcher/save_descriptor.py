import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.io import read_image
from extractor import ViTExtractor
from helpers import str2bool

# torchvision.set_video_backend("video_reader")


def parse():
    parser = argparse.ArgumentParser(
        description="Save descriptor with clicking on the image"
    )
    parser.add_argument(
        "-template",
        type=str,
        help="The template image from which to extract the descriptor",
    )
    parser.add_argument(
        "-descriptorname",
        type=str,
        help="Name of the descriptor to save in output/descritors/",
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
        "--layer", default=11, type=int, help="layer to create descriptors from."
    )

    args = parser.parse_args()
    return args


def get_descriptor(
    image_path_a: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
    descriptorname="testing",
    prefix_savepath="output/descriptors/",
):
    # extract descriptors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)

    descr = extractor.extract_descriptors(
        image_batch_a.to(device), layer, facet, bin, include_cls=False
    )

    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    # plot
    fig, axes = plt.subplots(1, 1)
    fig.suptitle("Double click to save descriptor, Right click to exit.")
    visible_patches = []
    radius = patch_size // 2

    # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
    axes.imshow(image_pil_a)
    pts = np.asarray(
        plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None)
    )

    while len(pts) == 1:
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
            visible_patches = []

        # draw chosen point
        center = (
            (x_descs_coor - 1) * stride + stride + patch_size // 2 - 0.5,
            (y_descs_coor - 1) * stride + stride + patch_size // 2 - 0.5,
        )
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        axes.add_patch(patch)
        visible_patches.append(patch)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        point_descriptor = descr[0, 0, raveled_desc_idx]

        pts = np.asarray(
            plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None)
        )
        os.makedirs(prefix_savepath, exist_ok=True)

        torch.save(point_descriptor, os.path.join(prefix_savepath, descriptorname))
        print("Saved descriptor to: ", os.path.join(prefix_savepath, descriptorname))

        exit()  # Do it only once for the release version so noone gets confused...


if __name__ == "__main__":
    args = parse()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(read_image(args.template).permute(1, 2, 0))
        cid = fig.canvas.mpl_connect("button_press_event", self.__onclick__)
        plt.show()
        return self.point

    load_size = 224
    bin = False
    stride = 4
    with torch.no_grad():
        descr = get_descriptor(
            args.template,
            load_size,
            args.layer,
            args.facet,
            bin,
            stride,
            args.model_type,
            args.descriptorname,
            prefix_savepath="output/descriptors/",
        )
