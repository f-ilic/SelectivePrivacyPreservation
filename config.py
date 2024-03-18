import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Training
parser.add_argument("--datasetname", type=str)
parser.add_argument("--weights_path", type=str, default=None)
parser.add_argument("--num_epochs", type=int, default=80)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_frames", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("-pretrained", action="store_true")
parser.add_argument("-train_backbone", action="store_true")
parser.add_argument("-privacy", action="store_true")  # Default is ACTION!
# parser.add_argument("-gpu", action="store_true")
parser.add_argument("--gpus", nargs="+", type=int, default=[])
parser.add_argument("--accumulate_grad_batches", type=int, default=1)


# Evaluation
parser.add_argument("--masked", default=None)  # person / background
parser.add_argument("--downsample", type=int, default=None)
parser.add_argument("--interpolation", default=None)  # nearest / bilinear / ...
parser.add_argument("--blur", default=None)  # weak / strong / None
parser.add_argument("--afd_combine_level", type=int, default=None)
parser.add_argument("-combine_masked", action="store_true")
parser.add_argument("--downsample_masked", type=int, default=None)
parser.add_argument("--interpolation_masked", default=None)  # nearest / bilinear / ...
parser.add_argument("-mean_fill", action="store_true")

# Our method
parser.add_argument("-selectively_mask", action="store_true")
parser.add_argument("--obfuscate", nargs="+", type=str, default=[])
parser.add_argument("-iid", action="store_true")

cfg = dict()


def build_cfg():
    args = parser.parse_args()
    cfg = args.__dict__.copy()
    print(f"-----------------------------------\n")
    print(f"Running Config:")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print(f"-----------------------------------\n")
    return cfg
