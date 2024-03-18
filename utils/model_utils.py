import random
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from dataset.db_stats import db_stats
from transforms.transform import DownsampleInterpolateUpsample
from models.valid_models import valid_models
import torch.nn as nn
from torchvision.models import resnet50, vgg16, resnet18, resnet101
from torchvision.models.vision_transformer import vit_b_16, vit_b_32


def set_requires_grad(model, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag


def load_weights(model, optimizer, weights_path, verbose=True):
    if weights_path != None:
        pth = torch.load(weights_path)
        model.load_state_dict(pth["state_dict"], strict=True)

        if verbose:
            print(f"Load Model from: {weights_path}")

        if "name" in pth:
            model.name = pth["name"]

        if optimizer != None:
            if verbose:
                print(f"Load Optimizer from: {weights_path}")
            optimizer.load_state_dict(pth["optimizer"])
            # hacky garbage from https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        if verbose:
            print(f"Nothing to Load")


def set_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def build_model_name(cfg):
    sample_rate = valid_models[cfg["architecture"]]["sample_rate"]
    num_frates = valid_models[cfg["architecture"]]["num_frames"]
    name = f"{cfg['architecture']}__{num_frates}x{sample_rate}"
    for k, v in cfg.items():
        if k in ["pretrained", "train_backbone"]:
            name += f"____{k}__{v}"
    return name


def build_model_name_singleframe(cfg):
    name = f"{cfg['architecture']}"
    for k, v in cfg.items():
        if k in ["pretrained", "train_backbone"]:
            name += f"____{k}__{v}"
    return name


def build_info_name(cfg):
    added_info = ""

    if cfg["obfuscate"] != []:
        added_info += f"__obfuscate_{'_'.join(cfg['obfuscate'])}"

    if cfg["masked"] != None:
        added_info += f"_masked_{cfg['masked']}"

    if cfg["downsample"] != None:
        added_info += f"__downsample{cfg['downsample']}"

    if cfg["interpolation"] != None:
        added_info += f"__interpolation_{cfg['interpolation']}"

    if cfg["blur"] != None:
        added_info += f"__blur_{cfg['blur']}"

    if cfg["downsample_masked"] != None:
        added_info += f"__downsample_masked_{cfg['downsample_masked']}x"

    if cfg["interpolation_masked"] != None:
        added_info += f"__interpolation_masked_{cfg['interpolation_masked']}"

    if cfg["afd_combine_level"] != None:
        added_info += f"__afd_combine_level_{cfg['afd_combine_level']}"

    if cfg["iid"] is True:
        added_info += f"__iid"
    return added_info


# This has all the strings of the torch hub video models.
# https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md


def build_model_privacy(architecture, pretrained, num_classes, train_backbone):
    if architecture == "resnet101":
        model = resnet101(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(2048, num_classes)

    elif architecture == "resnet50":
        model = resnet50(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(2048, num_classes)

    elif architecture == "resnet18":
        model = resnet18(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(512, num_classes)

    elif architecture == "vgg16":
        model = vgg16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif architecture == "vit":
        model = vit_b_16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Linear(768, num_classes)

    elif architecture == "vit_b_32":
        model = vit_b_32(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Linear(768, num_classes)
    else:
        raise ValueError("unsupported architecture")

    return model


def build_model(architecture, pretrained, num_classes, train_backbone):
    if architecture == "fast_r50":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
        )
        set_requires_grad(model, train_backbone)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
        print(model)
        # and then route the slow output into the nirvana.
        print("printing trainable parameters")

    elif architecture == "mvit_base_16x4":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", architecture, pretrained=pretrained
        )
        set_requires_grad(model, train_backbone)
        in_features = model.head.proj.in_features
        model.head.proj = nn.Linear(in_features, num_classes)
        print("done")

    elif "e2s_x3d" in architecture:
        model = E2SX3D(
            rgb_arch=architecture,
            flow_arch=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            train_rgbbackbone=train_backbone,
            train_flowbackbone=train_backbone,
        )

    elif architecture in valid_models.keys():
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", architecture, pretrained=pretrained
        )
        in_features = model.blocks[-1].proj.in_features
        set_requires_grad(model, train_backbone)
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
    else:
        raise ValueError("Unknown architecture.")
    return model


class E2SX3D(nn.Module):
    def init_backbone(self, arch, num_classes, pretrained, train_backbone):
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            arch.replace("e2s_", ""),
            pretrained=pretrained,
        )
        in_features = model.blocks[-1].proj.in_features
        set_requires_grad(model, train_backbone)
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
        return model

    def __init__(
        self,
        rgb_arch,
        flow_arch,
        num_classes,
        pretrained,
        train_rgbbackbone,
        train_flowbackbone,
    ):
        super(E2SX3D, self).__init__()
        self.rgb_arch = rgb_arch
        self.flow_arch = flow_arch
        self.rgbstream = self.init_backbone(
            rgb_arch, num_classes, pretrained, train_rgbbackbone
        )
        self.flowstream = self.init_backbone(
            flow_arch, num_classes, pretrained, train_flowbackbone
        )
        self.head = nn.Linear(2 * num_classes, num_classes)

    def forward(self, rgbs, flows):
        fs = self.flowstream(flows)
        aps = self.rgbstream(rgbs)
        x = torch.cat([fs, aps], dim=1)
        x = self.head(x)
        return x


def masking(video, cfg):
    if cfg["masked"] != None:
        raw = video["video"]
        mask = video["mask"]
        afd = video["afd"]

        if cfg["afd_combine_level"] != None:
            min_len = min(raw.shape[1], mask.shape[1], afd.shape[1])
            raw = raw[:, :min_len]
            mask = mask[:, :min_len]
            afd = afd[:, :min_len]
        else:
            min_len = min(raw.shape[1], mask.shape[1])
            raw = raw[:, :min_len]
            mask = mask[:, :min_len]

        mask = mask > 0

        # mean = torch.Tensor(db_stats[cfg["datasetname"]]["mean"])
        background = raw * (~mask)

        person = raw * mask
        if cfg["mean_fill"] == True:
            for i in range(3):  # each channel independently
                tmp = person[i, ...]
                mean = tmp.sum() / tmp.count_nonzero()
                background[i, ...] = background[i, ...].clip(
                    mean, background[i, ...].max()
                )

        if cfg["masked"] == "person":
            v1 = person
            v2 = background
            if cfg["afd_combine_level"] != None:
                v3 = afd * (mask)
        elif cfg["masked"] == "background":
            v1 = background
            v2 = person
            if cfg["afd_combine_level"] != None:
                v3 = afd * (~mask)

        if cfg["combine_masked"] == True:
            if cfg["downsample_masked"] != None:
                v1 = DownsampleInterpolateUpsample(
                    cfg["downsample_masked"],
                    cfg["interpolation_masked"],
                )(v1)

            if cfg["afd_combine_level"] != None:
                v1 = torch.lerp(
                    v1,
                    v3,
                    cfg["afd_combine_level"] / 100,
                )

            ret_video = v1 + v2
        else:
            ret_video = v2
        video["video"] = ret_video  # only change "video", leave the rest like it is.
    return video
