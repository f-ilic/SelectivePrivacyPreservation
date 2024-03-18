import os
from models.valid_models import action_models
from models.valid_models import privacy_models
from tqdm import tqdm

from itertools import chain, combinations


""" 
How does the masking of different attributes contribute the the action / privacy performance?
"""

""" Part of Fig.5 Caption:
Obfuscation with a Single Attribute and the Impact on Performance. 
Attribute importance is dataset dependent. For example,
notice how the ’Hand’ template contributes to a large decrease in
action recognition performance on IPN,as the action is determined soley
by the hand, whereas on SBU it does not.
"""


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


attributes = [
    "arm",
    "cheek",
    "eyes",
    "forehead",
    "hair",
    "hand",
    "lips",
    "torso",
    "leg",
]

return_codes = []
for i, combo in tqdm(
    enumerate(powerset(attributes), 1), total=len(list(powerset(attributes)))
):
    if len(combo) == 0:  # skip empty set
        continue

    # skip sets larger than 1
    if len(combo) > 1:
        continue

    att_args = " ".join([c for c in combo])

    for k in privacy_models.keys():
        run_str = f"CUDA_VISIBLE_DEVICES=0 python privacy_eval.py --architecture {k} --datasetname kth -pretrained --batch_size 32 --num_workers 16 -selectively_mask --obfuscate {att_args} -privacy"
        return_codes.append((os.system(run_str), run_str))

    for k in action_models.keys():
        run_str = f"CUDA_VISIBLE_DEVICES=0 python action_eval.py --architecture {k} --datasetname kth -pretrained -train_backbone --batch_size 2 --num_workers 16 -selectively_mask --obfuscate {att_args}"
        os.system(run_str)
        return_codes.append((os.system(run_str), run_str))


# print return codes tuple nicely formatted
for code, command in return_codes:
    if code != 0:
        error_string = "[ FAIL ]"
    else:
        error_string = "[ PASS ]"
    print(f"{error_string} --- {command}")
