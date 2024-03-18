from os.path import join
import pandas as pd
import shutil
from glob import glob
import random

root_dir = "data/kth/"

# list of all directories
dirs = sorted(glob(join(root_dir, "*")))

df = pd.DataFrame(columns=["video", "action", "person", "background", "frames", "set"])

actions = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
persons = [f"person{i:02d}" for i in range(1, 26)]
backgrounds = [f"d{i}" for i in range(1, 5)]

rows_list = []
array = ["train", "train", "train", "test"]
for p in persons:
    for a in actions:
        for d, split in zip(backgrounds, random.sample(array, len(array))):
            # for d, split in zip(backgrounds, array):
            path = f"{p}_{a}_{d}_uncomp"
            n_frames = len(glob(join(root_dir, path, "*.png")))
            rows_list.append(
                {
                    "video": path,
                    "action": a,
                    "person": p,
                    "background": d,
                    "frames": n_frames,
                    "set": split,
                }
            )

df = pd.DataFrame(rows_list)
df.to_csv(join("data", "splits", "kth", "metadata.csv"), index=False)

print(df)
