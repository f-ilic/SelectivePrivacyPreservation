import sys
from os.path import join

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import build_cfg, cfg, parser
from dataset.db_factory import DBfactory
from models.valid_models import valid_models
from simulation.simulation import Simulation
from utils.info_print import print_data_augmentation_transform, print_learnable_params
from utils.model_utils import build_model, build_model_name, set_seed
from utils.Trainer import Trainer, Trainer_E2SX3D

parser.add_argument("--architecture", type=str)
cfg = build_cfg()

import matplotlib

matplotlib.use("Agg")
# Setting reproducibility


def main():
    set_seed()
    torch.backends.cudnn.benchmark = True
    datasetname = cfg["datasetname"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    accum_every = cfg["accumulate_grad_batches"]
    gpus = cfg["gpus"]
    multigpu = len(gpus) > 1

    train_dataset = DBfactory(datasetname, set_split="train", config=cfg)
    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    # ----------------- Setup Model & Load weights if supplied -----------------
    architecture = cfg["architecture"]

    if architecture not in valid_models.keys():
        raise ValueError("This model is not defined in the valid_model dictionary.")

    model = build_model(
        architecture,
        cfg["pretrained"],
        train_dataset.num_classes,
        cfg["train_backbone"],
    )

    model.name = build_model_name(cfg)
    model.configuration = cfg
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    if multigpu:
        model = nn.DataParallel(model, device_ids=gpus)
        model.name = model.module.name

    # load_weights(model, optimizer, cfg["weights_path"])
    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    sim_name = f"action/{cfg['datasetname']}/{model.name}"
    best_top1 = 0

    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        print(f"-------------- CFG ------------------\n")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        print(f"-------------------------------------\n")

        cfg["executed"] = f'python {" ".join(sys.argv)}'
        print(f'Running: {cfg["executed"]}\n\n\n')
        print_learnable_params(model, verbose=False)
        print_data_augmentation_transform(train_dataset.transform)
        print(f"Begin training: {model.name}")

        writer = SummaryWriter(join(sim.outdir, "tensorboard"))

        if "e2s_x3d" in cfg["architecture"]:
            trainer = Trainer_E2SX3D(sim)
        else:
            trainer = Trainer(sim)

        # -------------- MAIN TRAINING LOOP  ----------------------
        for epoch in range(cfg["num_epochs"]):
            trainer.do(
                "train",
                model,
                train_dataloader,
                epoch,
                criterion,
                optimizer,
                writer,
                log_video=False,
                accumulate_grad_batches=accum_every,
            )

            if epoch % 5 == 0 or epoch == cfg["num_epochs"] - 1:
                curr_top1 = trainer.do(
                    "test",
                    model,
                    test_dataloader,
                    epoch,
                    criterion,
                    None,
                    writer,
                    log_video=False,
                )
                if curr_top1 > best_top1:
                    best_top1 = curr_top1

                    checkpoint = {
                        "epoch": epoch,
                        "state_dict": (
                            model.module.state_dict()
                            if multigpu
                            else model.state_dict()
                        ),
                        "optimizer": optimizer.state_dict(),
                    }
                    sim.save_pytorch(checkpoint, epoch=epoch)

        print(f"\nRun {sim.outdir} finished\n")

        writer.close


if __name__ == "__main__":
    main()
