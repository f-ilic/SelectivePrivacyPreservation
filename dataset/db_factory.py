from matplotlib import pyplot as plt

from config import build_cfg
from dataset.ipn import IPN
from dataset.kth import KTH
from dataset.sbu import SBU
from transforms.transform import InverseNormalizeVideo, transform_default
from utils.VideoTensorViewer import VideoTensorViewer


def DBfactory(dbname, set_split, config):
    if dbname in ["ipn"]:
        T = transform_default(config, set_split)
        T_inv = InverseNormalizeVideo(config)
        data_root = f"data/{dbname}"
        annot_root = f"data/splits/ipn/"
        db = IPN(data_root, annot_root, set_split, T, T_inv, config)
    elif dbname in ["kth"]:
        T = transform_default(config, set_split)
        T_inv = InverseNormalizeVideo(config)
        data_root = f"data/{dbname}"
        annot_root = f"data/splits/kth/"
        db = KTH(data_root, annot_root, set_split, T, T_inv, config)
    elif dbname in ["sbu"]:
        T = transform_default(config, set_split)
        T_inv = InverseNormalizeVideo(config)
        data_root = f"data/{dbname}"
        annot_root = f"data/splits/sbu/"
        db = SBU(data_root, annot_root, set_split, T, T_inv, config)
    else:
        raise ValueError(f"Invalid Database name {dbname}")
    return db


def show_single_videovolume():
    cfg = build_cfg()
    cfg["num_frames"] = 30
    cfg["sampling_rate"] = 1
    cfg["architecture"] = "x3d_s"

    # cfg = None
    dl = DBfactory(cfg["datasetname"], set_split="train", config=cfg)

    for step, (s, masks, flows, labels) in enumerate(dl):
        sample = dl.inverse_normalise(s)
        # VideoTensorViewer(torch.stack([sample, masks], dim=3))
        VideoTensorViewer(sample)
        plt.show(block=True)


if __name__ == "__main__":
    show_single_videovolume()
