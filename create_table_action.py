import os
import numpy as np
from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pandas as pd

sns.set_style("whitegrid")
sns.set_style("darkgrid")


df_all = pd.DataFrame()

# datasets = ["kth", "ipn", "sbu"]
datasets = ["sbu"]

for dataset in datasets:
    base_dir = f"runs/action_eval/{dataset}/"

    tbfiles = []
    for filename in glob.glob(f"{base_dir}/**/events.out.tfevents.*", recursive=True):
        tbfiles.append(filename)

    for tbfile in tbfiles:
        name = tbfile.replace(base_dir, "").split("/")[1].split("__")[0]
        experiment = tbfile.replace(base_dir, "").split("/")[0]
        experiment = experiment.replace(
            "____pretrained__True____train_backbone__True", ""
        )

        experiment = experiment.replace("_", " ")
        experiment = experiment.replace("person", "person\n")
        experiment = experiment.replace("background", "background\n")
        experiment = experiment.replace("downsample masked", "downsample")
        experiment = experiment.replace("afd combine level", "afd level")
        experiment = experiment.replace("  ", " ")
        reader = SummaryReader(tbfile)
        df = reader.scalars
        df["name"] = name
        df["experiment"] = experiment
        df["dataset"] = dataset

        print(f"{dataset} \t {name} \t {experiment}")

        t = df[df.tag == "test/top1"]
        # t = t.loc[t["value"].idxmax()]
        t = t.loc[t["value"]]

        bestepoch = (
            "/".join(tbfile.split("/")[:-2]) + f"/models/checkpoint_epoch{t.step}.pt"
        )
        bestpt = "/".join(tbfile.split("/")[:-3]) + "/best.pt"

        t["path"] = bestepoch
        if not os.path.exists(bestepoch):
            bestepoch = (
                "/".join(tbfile.split("/")[:-2])
                + f"/models/checkpoint_epoch{t.step+5}.pt"
            )

        if os.path.exists(bestepoch):
            os.system(f"cp {bestepoch} {bestpt}")
            print(f"\t \t {t.step} -> best.pt")
        else:
            print(f"{bestepoch} does not exist")

        df_all = pd.concat([df_all, pd.DataFrame([dict(t)])])

# tmp.loc[tmp['value'].idxmax()]
tmp = df_all[(df_all.tag == "test/top1")]
individual = tmp.groupby(["experiment", "dataset"]).max()[
    "value"
]  # for each experiment, get the max value for each dataset

averages = (
    tmp.groupby(["experiment", "dataset"]).max()["value"].groupby("dataset").mean()
)  # mean over datasets


LATEX_individual = pd.pivot_table(
    pd.DataFrame(individual), index="experiment", columns="dataset", values="value"
)

averages = (
    tmp.groupby(["experiment", "dataset"]).max()["value"].groupby("dataset").mean()
)
LATEX_averages = pd.pivot_table(
    pd.DataFrame(averages), columns="dataset", values="value"
)

print(
    LATEX_individual.to_latex(
        formatters={"name": str.upper},
        float_format="{:.2f}".format,
    )
)

print(
    LATEX_averages.to_latex(
        formatters={"name": str.upper},
        float_format="{:.2f}".format,
    )
)
