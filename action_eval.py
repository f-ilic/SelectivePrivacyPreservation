from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import build_cfg, cfg, parser
from dataset.db_factory import DBfactory
from models.valid_models import valid_models
from simulation.simulation import Simulation
from utils.model_utils import (
    build_info_name,
    build_model,
    build_model_name,
    load_weights,
    set_seed,
)
from utils.Trainer import Trainer, Trainer_E2SX3D

parser.add_argument("--architecture", type=str)
cfg = build_cfg()


# Setting reproducibility


def main():
    set_seed()
    torch.backends.cudnn.benchmark = True
    datasetname = cfg["datasetname"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    # ----------------- Setup Model & Load weights if supplied -----------------
    architecture = cfg["architecture"]

    if architecture not in valid_models.keys():
        raise ValueError("This model is not defined in the valid_model dictionary.")

    model = build_model(
        architecture,
        cfg["pretrained"],
        test_dataset.num_classes,
        cfg["train_backbone"],
    )

    model.name = build_model_name(cfg)
    model.configuration = cfg

    print(
        "MAKE SURE YOU RUN create_table_action.py FIRST SO THAT ALL THE BEST.PT FILES ARE CREATED"
    )
    cfg["weights_path"] = f"runs/action/{datasetname}/{model.name}/best.pt"

    load_weights(model, None, cfg["weights_path"])

    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    added_info = build_info_name(cfg)

    # sim_name = (
    #     f"action_eval/{datasetname}/{cfg['datasetname']}{added_info}/{model.name}"
    # )
    sim_name = f"action_eval_attributes/{datasetname}/{cfg['datasetname']}{added_info}/{model.name}"

    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        writer = SummaryWriter(join(sim.outdir, "tensorboard"))
        if "e2s_x3d" in cfg["architecture"]:
            trainer = Trainer_E2SX3D(sim)
        else:
            trainer = Trainer(sim)

        trainer.do(
            "eval",
            model,
            test_dataloader,
            0,
            criterion,
            None,
            writer,
            log_video=True,
        )

        writer.close


if __name__ == "__main__":
    main()
    print("\n\n [ DONE ]")
