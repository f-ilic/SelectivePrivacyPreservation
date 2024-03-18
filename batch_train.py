import itertools
import os
from models.valid_models import valid_models
from models.valid_models import action_models
from models.valid_models import privacy_models

for k in privacy_models.keys():
    for db in ["kth", "ipn", "sbu"]:
        for train_backbone in ["", "-train_backbone"]:
            to_run = f"CUDA_VISIBLE_DEVICES=0 python privacy_train.py --architecture {k} --datasetname {db} -pretrained {train_backbone} --batch_size 128 --num_epochs 500 --lr 3e-4 -privacy --num_workers 16"
            # print(to_run)
            os.system(to_run)
