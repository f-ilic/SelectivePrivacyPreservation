import json
import sys
from email import parser
from os.path import join

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import build_cfg, cfg, parser
from dataset.db_factory import DBfactory
from simulation.simulation import Simulation
from utils.info_print import *
from utils.model_utils import (
    build_info_name,
    build_model_name_singleframe,
    build_model_privacy,
    load_weights,
)
from utils.Trainer import Trainer

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--architecture", type=str)
cfg = build_cfg()


def main():
    torch.backends.cudnn.benchmark = True

    datasetname = cfg["datasetname"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    accum_every = cfg["accumulate_grad_batches"]
    gpus = cfg["gpus"]
    multigpu = len(gpus) > 1

    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    # ----------------- Setup Model & Load weights if supplied -----------------

    model = build_model_privacy(
        cfg["architecture"],
        cfg["pretrained"],
        test_dataset.num_classes,
        cfg["train_backbone"],
    )
    model.name = build_model_name_singleframe(cfg)

    if multigpu:
        model = nn.DataParallel(model, device_ids=gpus)
        model.name = model.module.name

    optimizer = optim.AdamW(model.parameters())
    print(
        "MAKE SURE YOU RUN create_table_action.py FIRST SO THAT ALL THE BEST.PT FILES ARE CREATED"
    )
    cfg["weights_path"] = f"runs/privacy/{datasetname}/{model.name}/best.pt"
    load_weights(model, optimizer, cfg["weights_path"])

    criterion = nn.CrossEntropyLoss().cuda()

    model = model.cuda()
    added_info = build_info_name(cfg)

    sim_name = f"privacy_eval_attributes/{datasetname}/{cfg['datasetname']}{added_info}/{model.name}"
    # sim_name = (
    #     f"privacy_eval/{datasetname}/{cfg['datasetname']}{added_info}/{model.name}"
    # )

    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        print(f'Running: python {" ".join(sys.argv)}\n\n\n')
        print_learnable_params(model)
        print_data_augmentation_transform(test_dataset.transform)
        print(f"Begin training: {model.name}")

        with open(join(sim.outdir, "cfg.txt"), "w") as f:
            json.dump(cfg, f, indent=2)

        writer = SummaryWriter(join(sim.outdir, "tensorboard"))
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
        print(f"\nRun {sim.outdir} finished\n")

        writer.close


if __name__ == "__main__":
    main()
