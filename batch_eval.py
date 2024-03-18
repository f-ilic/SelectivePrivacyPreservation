import itertools
import os
from models.valid_models import valid_models
from models.valid_models import action_models
from models.valid_models import privacy_models


"""
Skeleton on how to evaluate the runs you want to compare.
The tensorboard files then contain all the necessary information that
you have to parse out with the scripts in the `analysis` directory.
"""


for k in action_models.keys():
    for db in ["ipn", "kth", "sbu"]:
        for downsample in ["16"]:
            run_string = f"CUDA_VISIBLE_DEVICES=0 python action_eval.py --architecture {k} --datasetname {db} -pretrained --batch_size 1 --downsample {downsample} --interpolation nearest"
            os.system(run_string)

for k in privacy_models.keys():
    for db in ["ipn", "kth", "sbu"]:
        for downsample in ["16"]:
            run_string = f"CUDA_VISIBLE_DEVICES=0 python privacy_eval.py --architecture {k} --datasetname {db} -pretrained --batch_size 1 --downsample {downsample} --interpolation nearest -privacy"
            os.system(run_string)
