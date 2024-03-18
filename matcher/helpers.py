import os

import numpy as np
import PIL
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from matcher.extractor import ViTExtractor


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y)
    """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Cannot convert to boolean")


# get the best matching descriptor in the first image of the video to the descriptor
def get_best_matching_descriptor(descriptor, image_path):
    with torch.no_grad():
        # extract descriptors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = ViTExtractor(device=device)
        image_batch, _ = extractor.preprocess(image_path, load_size=224)

        descs = extractor.extract_descriptors(
            image_batch.to(device), layer=11, facet="key", bin=False, include_cls=False
        )

        # compute similarity
        sim = chunk_cosine_sim(descriptor[None, None, None], descs)
        sim_image = sim.reshape(extractor.num_patches)
        sim_image = sim_image.cpu().numpy()

        # get best matching descriptor
        best_matching_descriptor = np.argmax(sim_image)
        return descs[:, :, best_matching_descriptor].squeeze()


def save_similarity_from_descriptor(
    descriptor,
    videoname: str,
    images: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
    prefix_savedir="output/similarities/",
    name=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    img_size = PIL.Image.open(images[0]).size[::-1]

    similarities = []

    for en, image_path_b in enumerate(images):
        print(f"Computing Descriptors {en}")
        image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
        descs_b = extractor.extract_descriptors(
            image_batch_b.to(device), layer, facet, bin, include_cls=False
        )
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        sim = chunk_cosine_sim(descriptor[None, None, None], descs_b)
        similarities.append(sim)

        sim_image = sim.reshape(num_patches_b)
        os.makedirs(prefix_savedir + f"/{name}_{videoname}", exist_ok=True)
        sim_image = transforms.Resize(img_size, antialias=True)(sim_image.unsqueeze(0))
        save_image(sim_image, f"{prefix_savedir}/{name}_{videoname}/{en:04d}.png")


def similarity_from_descriptor(
    descriptor,
    images: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    similarities = []
    img_size = PIL.Image.open(images[0]).size[::-1]

    ret = []
    for en, image_path_b in tqdm(enumerate(images), total=len(images), position=1):
        # print(f"Computing Descriptors {en}")
        image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
        descs_b = extractor.extract_descriptors(
            image_batch_b.to(device), layer, facet, bin, include_cls=False
        )
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        sim = chunk_cosine_sim(descriptor[None, None, None], descs_b)
        similarities.append(sim)

        sim_image = sim.reshape(num_patches_b)
        ret_img = transforms.Resize(img_size, antialias=True)(sim_image.unsqueeze(0))

        ret.append(ret_img.squeeze())
    return ret
