import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from RAFT.core.utils.utils import InputPadder
from torch.nn.functional import grid_sample
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

# torchvision.set_video_backend("video_reader")


def warp(img, flow, mode="bilinear", padding_mode="zeros"):
    # img.shape -> 3, H, W
    # flow.shape -> H, W, 2
    (
        _,
        h,
        w,
    ) = img.shape
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    f0 = torch.stack(torch.meshgrid(x_coords, y_coords)).permute(2, 1, 0)

    f = f0 + torch.stack([2 * (flow[:, :, 0] / w), 2 * (flow[:, :, 1] / h)], dim=2)
    warped = grid_sample(
        img.unsqueeze(0), f.unsqueeze(0), mode=mode, padding_mode=padding_mode
    )
    return warped.squeeze()


def batch_warp(img, flow, mode="bilinear", padding_mode="zeros"):
    # img.shape -> B, 3, H, W
    # flow.shape -> B, H, W, 2
    (
        b,
        c,
        h,
        w,
    ) = img.shape
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    f0 = (
        torch.stack(torch.meshgrid(x_coords, y_coords))
        .permute(2, 1, 0)
        .repeat(b, 1, 1, 1)
    )

    f = f0 + torch.stack([2 * (flow[..., 0] / w), 2 * (flow[..., 1] / h)], dim=3)
    warped = grid_sample(img, f, mode=mode, padding_mode=padding_mode)
    return warped.squeeze()


def upsample_flow(flow, h, w):
    # usefull function to bring the flow to h,w shape
    # so that we can warp effectively an image of that size with it
    h_new, w_new, _ = flow.shape
    flow_correction = torch.Tensor((h / h_new, w / w_new))
    f = flow * flow_correction[None, None, :]

    f = (
        Resize((h, w), interpolation=InterpolationMode.BICUBIC)(f.permute(2, 0, 1))
    ).permute(1, 2, 0)
    return f


def firstframe_warp(
    flows, interpolation_mode="nearest", usecolor=True, seed_image=None
):
    h, w, _ = flows[0].shape
    c = 3 if usecolor == True else 1

    t = len(flows)
    flows = torch.stack(flows)

    if usecolor:
        inits = (torch.rand((1, c, h, w))).repeat(t, 1, 1, 1)
    else:
        inits = (torch.rand((1, 1, h, w))).repeat(t, 3, 1, 1)

    if seed_image != None:
        inits = read_image(seed_image, ImageReadMode.RGB) / 255.0
        inits = Resize((h, w), interpolation=InterpolationMode.NEAREST)(inits)
        inits = inits.repeat(t, 1, 1, 1)

    warped = batch_warp(inits, flows, mode=interpolation_mode)
    masks = ~(warped.any(dim=1))
    masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)
    warped[masks] = inits[masks]
    warped = torch.cat([inits[0, ...].unsqueeze(dim=0), warped], dim=0)
    warped = (warped).clip(0, 1)
    # warped = Resize(size=(int(h * 0.5), int(w * 0.5)))(warped)
    # warped = warped[:, :, 10:-10, 10:-10]
    warped = (warped.permute(0, 2, 3, 1) * 255).float()
    return warped


def continuous_warp(flows, interpolation_mode="nearest", usecolor=True):
    h, w, _ = flows[0].shape
    t = len(flows)
    c = 3 if usecolor == True else 1

    init = torch.rand((c, h, w))
    warp_i = init
    l = []
    for en, flow in enumerate(flows):
        warp_i = warp(warp_i, flow, mode=interpolation_mode, padding_mode="reflection")
        if c == 1:
            warp_i = warp_i.unsqueeze(0)

        mask = ~(warp_i.any(dim=0))
        mask = mask.repeat(c, 1, 1)
        warp_i[mask] = init[mask]
        l.append(warp_i)

    warped = torch.stack(l, axis=0)
    warped = (warped).clip(0, 1)
    warped = (warped.permute(0, 2, 3, 1) * 255).float()
    return warped


def flow_from_video(model, frames_list, upsample_factor=1):
    num_frames = len(frames_list)
    all_flows = []
    c, h, w = frames_list[0].shape
    with torch.no_grad():
        for i in range(num_frames - 1):
            image0 = frames_list[i].unsqueeze(0).cuda()
            image1 = frames_list[i + 1].unsqueeze(0).cuda()

            if upsample_factor != 1:
                image0 = nn.Upsample(scale_factor=upsample_factor, mode="bilinear")(
                    image0
                )
                image1 = nn.Upsample(scale_factor=upsample_factor, mode="bilinear")(
                    image1
                )

            padder = InputPadder(image0.shape)
            image0, image1 = padder.pad(image0, image1)
            _, flow_up = model(image0, image1, iters=12, test_mode=True)
            flow_output = (
                padder.unpad(flow_up).detach().cpu().squeeze().permute(1, 2, 0)
            )
            fl = flow_output.detach().cpu().squeeze()
            fl = upsample_flow(fl, h, w)
            all_flows.append(fl)
    return all_flows


def gather_videopaths_in_directory(directory, alphabetical_range=["A", "Z"]):
    video_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            curr_file = os.path.join(path, name)
            if curr_file.endswith(".avi"):
                first_letter_of_class = name.split("_")[1][0]
                if (
                    first_letter_of_class >= alphabetical_range[0]
                    and first_letter_of_class <= alphabetical_range[1]
                ):
                    video_list.append(curr_file)
    return video_list


def set_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def save_tensorvideo_as_images(path, video):
    for t in range(video.shape[0]):
        img = (video[t, ...] * 255).byte()
        cv2.imwrite(path + ".png", img.numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
