import argparse
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision.io import read_image
from torchvision.utils import make_grid
from matcher.helpers import similarity_from_descriptor
from matcher.helpers import get_best_matching_descriptor

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--descriptor", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    images = [args.image]
    descriptor = torch.load(args.descriptor)
    # targeted_descr = get_best_matching_descriptor(descriptor, images[0])

    with torch.no_grad():
        hair = similarity_from_descriptor(descriptor, images)[0].cpu().squeeze() * 255.0
    orig = read_image(images[0]).squeeze().permute(1, 2, 0)

    hair = hair.clamp(80, 255)

    # # split into 24x24 patches
    new = orig.unfold(0, 48, 48).unfold(1, 48, 48)
    new = rearrange(new, "x y c h w  -> (x y) c h w")
    plotable = make_grid(new, nrow=6, padding=10, pad_value=255)
    plt.figure()
    plt.axis("off")
    plt.imshow(plotable.permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    # plt.savefig("readme_assets/orig_tiled.png", dpi=300, bbox_inches="tight")
    # print("Saved image to readme_assets/orig_tiled.png")

    # split into 24x24 patches
    new = hair.unfold(0, 48, 48).unfold(1, 48, 48)
    new = rearrange(new, "x y h w  -> (x y) h w").unsqueeze(1)
    plotable = make_grid(new, nrow=6, padding=10, pad_value=255)
    plt.axis("off")
    plt.imshow(plotable.permute(1, 2, 0)[:, :, 0], cmap="hot")
    plt.tight_layout()
    plt.show()
    # plt.savefig("readme_assets/hair_tiled.png", dpi=300, bbox_inches="tight")
    # print("Saved image to readme_assets/hair_tiled.png")
