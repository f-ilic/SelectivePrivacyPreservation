import sys
import torch
from os.path import join

from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from config import cfg, parser, build_cfg
from dataset.db_factory import DBfactory
from utils.Trainer import Trainer, Trainer_E2SX3D
from simulation.simulation import Simulation
from utils.info_print import print_data_augmentation_transform, print_learnable_params
from utils.model_utils import load_weights
from transforms.transform import valid_models
from utils.model_utils import (
    build_model,
    build_model_name,
    load_weights,
    set_seed,
)

# from train_video_architecture import build_model, build_model_name

parser.add_argument("--architecture", type=str)
parser.add_argument("--use_motion_aligned", default="FULL", type=str)

cfg = build_cfg()


# Setting reproducibility


def main():
    set_seed()
    torch.backends.cudnn.benchmark = True
    datasetname = cfg["datasetname"]
    batch_size = cfg["batch_size"]
    num_workers = 16

    test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

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
    load_weights(model, None, cfg["weights_path"])

    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    sim_name = f"ablate_afdnoise/{cfg['datasetname']}/{model.name}"
    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        cfg["executed"] = f'python {" ".join(sys.argv)}'
        print(f'Running: {cfg["executed"]}\n\n\n')
        print_learnable_params(model)
        print_data_augmentation_transform(test_dataset.transform)
        print(f"Begin training: {model.name}")

        writer = SummaryWriter(join(sim.outdir, "tensorboard"))
        if "e2s_x3d" in cfg["architecture"]:
            trainer = Trainer_E2SX3D(sim)
        else:
            trainer = Trainer(sim)

        # -------------- MAIN TRAINING LOOP  ----------------------
        for noise_level in torch.arange(0, 110, 10):
            cfg["afd_combine_level"] = noise_level / 100.0
            # if "e2s_x3d" in cfg["architecture"] and noise_level < 50:
            #     cfg["afd_combine_level"] = torch.sigmoid(noise_level) / 100.0
            print(f'{cfg["afd_combine_level"]:0.2f}')

            # Re-init the dataset every time with the correct noise level
            test_dataset = DBfactory(datasetname, set_split="test", config=cfg)

            test_dataloader = DataLoader(
                test_dataset,
                batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=False,
            )

            trainer.do(
                "ablate_afdnoise",
                model,
                test_dataloader,
                noise_level,
                criterion,
                None,
                writer,
                log_video=True,
            )

        writer.close


if __name__ == "__main__":
    main()
