import os

for descr in [
    "arm",
    "cheek",
    "eyes",
    "forehead",
    "hair",
    "hand",
    "lips",
    "torso",
    "leg",
]:
    DRY_RUN = True

    print(f"{'*'*20} WORKING ON: {descr} {'*'*20}")
    run_str = f"CUDA_VISIBLE_DEVICES=0 python similarity_from_template_for_dataset.py -descriptorpath output/descriptors/{descr}.pt --use_targeted False"

    if DRY_RUN:
        print(run_str)
    else:
        os.system(run_str)
