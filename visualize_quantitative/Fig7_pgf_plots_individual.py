import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from os import path, listdir
from os.path import join
from glob import glob
import random
import torch

from utils.model_utils import set_seed

import matplotlib as mpl

mpl.use("Agg")

if __name__ == "__main__":

    def f(lam, a, p):
        return (1 - lam) * a + lam * (1 - p)

    labels = [
        "BDQ \cite{bqn}",
        "ALF \cite{wu2020privacy}",
        "ELR\cite{ryoo2018extremelowres} s=2",
        "ELR\cite{ryoo2018extremelowres} s=4",
        "ELR\cite{ryoo2018extremelowres} s=8",
        "ELR\cite{ryoo2018extremelowres} s=16",
        "ELR\cite{ryoo2018extremelowres} s=32",
        "ELR\cite{ryoo2018extremelowres} s=64",
        "Ours \\textdagger",
        "Ours",
    ]
    ipn_action = [81, 76, 82.31, 81.76, 79.48, 70.82, 52.96, 31.63, 87.11, 83.15]
    ipn_privacy = [59, 65, 80.04, 72.01, 70.08, 64.32, 63.29, 62.7, 55.38, 54.12]

    kth_action = [91.11, 85.89, 91.64, 92.99, 91.22, 91.22, 85.57, 56.21, 88.67, 82.70]
    kth_privacy = [7.15, 19.27, 91.82, 92.50, 91.58, 88.86, 82.56, 58.35, 5.46, 4.31]

    sbu_action = [84.04, 82.00, 97.93, 98.27, 98.47, 96.27, 92.42, 80.05, 84.04, 86.74]
    sbu_privacy = [34.18, 48.00, 85.1, 91.48, 84.04, 82.97, 64.89, 43.61, 11.7, 13.19]

    # for lbl, a, p in zip(labels, ipn_action, ipn_privacy):
    # for lbl, a, p in zip(labels, kth_action, kth_privacy):
    for lbl, a, p in zip(labels, sbu_action, sbu_privacy):
        if lbl in [
            "ELR\cite{ryoo2018extremelowres} s=2",
            "ELR\cite{ryoo2018extremelowres} s=4",
            "ELR\cite{ryoo2018extremelowres} s=8",
            "ELR\cite{ryoo2018extremelowres} s=16",
            # "ELR\cite{ryoo2018extremelowres} s=32",
            # "ELR\cite{ryoo2018extremelowres} s=64",
        ]:
            continue
        lam = np.linspace(0, 1, 2)

        if lbl in ["Ours \\textdagger", "Ours"]:
            print("\\addplot+[style={mark=square*,ultra thick}]")
        else:
            print("\\addplot+[]")
        print("coordinates {")
        for l in lam:
            y = f(l, a / 100, p / 100)
            print(f"({100*l:.2f}, {100*y:.2f})", end="")
        print("};")
        print("\\addlegendentry{%s}" % lbl)
        print()
