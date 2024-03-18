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

    train_dataset = DBfactory(datasetname, set_split="train", config=cfg)

    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

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
        train_dataset.num_classes,
        cfg["train_backbone"],
    )
    model.name = build_model_name_singleframe(cfg)

    if multigpu:
        model = nn.DataParallel(model, device_ids=gpus)
        model.name = model.module.name

    optimizer = optim.AdamW(model.parameters())
    load_weights(model, optimizer, cfg["weights_path"])

    criterion = nn.CrossEntropyLoss().cuda()

    model = model.cuda()

    sim_name = f"privacy/{datasetname}/{model.name}"
    best_top1 = 0
    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        print(f'Running: python {" ".join(sys.argv)}\n\n\n')
        print_learnable_params(model)
        print_data_augmentation_transform(train_dataset.transform)
        print(f"Begin training: {model.name}")

        with open(join(sim.outdir, "cfg.txt"), "w") as f:
            json.dump(cfg, f, indent=2)

        writer = SummaryWriter(join(sim.outdir, "tensorboard"))
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

        trainer.do(
            "test",
            model,
            test_dataloader,
            epoch,
            criterion,
            None,
            writer,
        )
        print(f"\nRun {sim.outdir} finished\n")

        writer.close


if __name__ == "__main__":
    main()
