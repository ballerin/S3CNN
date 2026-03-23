import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#from matplotlib import cm

import pickle
from tqdm.auto import tqdm
#from tqdm import tqdm
import time

DATASET_SIZES = 10

PLOT_DATA = {
    "CNN_NR_NR":    True,
    "CNN_NR_R":     True,
    "CNN_R_R":      False,
    "SO3_NR_NR":    True,
    "SO3_NR_R":     True,
    "SO3_R_R":      False,
    "SPINS_NR_NR":    True,
    "SPINS_NR_R":     True,
    "SPINS_R_R":      False,
}

CNN_NR = ["2025-11-11-bn-32",]*10

CNN_R = ["2026-02-11", #1
         "2026-02-11", #2
         "2026-02-11", #3
         "2026-02-19-test1", #4
         "2026-02-11", #5
         "2026-02-19-test1", #6
         "2026-02-11", #7
         "2026-02-11", #8
         "2026-02-19-test1", #9
         "2026-02-11",] #10

SO3_NR = ["2025-11-19-lr-ln",]*10

SO3_R = ["2025-11-19-lr-ln",]*10

SPINS_NR = ["2026-03-07-mag-ln-32",]*10

SPINS_R = ["2026-03-07-mag-ln-32",]*10

if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([1.5,2.4])
    ax.set_xticks(list(range(1,11)))
    #plt.title("Wind to Wind prediction by dataset size")
    plt.xlabel("Training dataset size (# of years)")
    plt.ylabel("Performance (lower is better)")

    # Color and linestyle mapping from legend
    COLORS = {
        "CNN": "#1b9e77",
        "sCNN": "#d95f02",
        "GsCNN": "#7570b3",
    }
    LINESTYLES = {
        "NR/NR": '-',
        "NR/R": '--',
        "R/R": ':',
    }

    def get_mean_loss(file):
        return np.mean(pickle.load(open(file, "rb")))

    def get_cnn_path(ds, run, rb=False):
        if rb:
            return f"experiments/wind->wind/CNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/wind->wind/CNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_cnn_r_path(ds, run):
        return f"experiments/wind->wind/CNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_so3_path(ds, run, rb=False):
        if rb:
            return f"experiments/wind->wind/SO3/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/wind->wind/SO3/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_so3_r_path(ds, run):
        return f"experiments/wind->wind/SO3/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_spins_path(ds, run, rb=False):
        if rb:
            return f"experiments/wind->wind/SPINS/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/wind->wind/SPINS/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_spins_r_path(ds, run):
        return f"experiments/wind->wind/SPINS/RB/{ds+1}/{run}/RB_test_losses.pckl"

    if PLOT_DATA["CNN_NR_NR"]:
        mean_losses_CNN_NR_NR = [get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_NR, color=COLORS["CNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["CNN_NR_R"]:
        mean_losses_CNN_NR_R = [get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_R, color=COLORS["CNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["CNN_R_R"]:
        mean_losses_CNN_R_R = [get_mean_loss(get_cnn_r_path(ds, CNN_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_R_R, color=COLORS["CNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["SPINS_NR_NR"]:
        mean_losses_SPINS_NR_NR = [get_mean_loss(get_spins_path(ds, SPINS_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_NR_NR, color=COLORS["sCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SPINS_NR_R"]:
        mean_losses_SPINS_NR_R = [get_mean_loss(get_spins_path(ds, SPINS_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_NR_R, color=COLORS["sCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SPINS_R_R"]:
        mean_losses_SPINS_R_R = [get_mean_loss(get_spins_r_path(ds, SPINS_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_R_R, color=COLORS["sCNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["SO3_NR_NR"]:
        mean_losses_SO3_NR_NR = [get_mean_loss(get_so3_path(ds, SO3_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_NR_NR, color=COLORS["GsCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SO3_NR_R"]:
        mean_losses_SO3_NR_R = [get_mean_loss(get_so3_path(ds, SO3_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_NR_R, color=COLORS["GsCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SO3_R_R"]:
        mean_losses_SO3_R_R = [get_mean_loss(get_so3_r_path(ds, SO3_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_R_R, color=COLORS["GsCNN"], linestyle=LINESTYLES["R/R"])

    plt.legend(
        [
            "CNN NR/NR",
            "CNN NR/R",
            "sCNN NR/NR",
            "sCNN NR/R",
            "GsCNN NR/NR",
            "GsCNN NR/R",
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.savefig("figures/windwind_earlyresults_dataset_size.png", bbox_inches='tight')