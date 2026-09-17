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
    "S3CNN_NR_NR":    True,
    "S3CNN_NR_R":     True,
    "S3CNN_R_R":      False,
    "SWSCNN_NR_NR":    True,
    "SWSCNN_NR_R":     True,
    "SWSCNN_R_R":      False,
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

S3CNN_NR = ["2025-11-19-lr-ln",]*10

S3CNN_R = ["2025-11-19-lr-ln",]*10

SWSCNN_NR = ["2026-03-07-mag-ln-32",]*10

SWSCNN_R = ["2026-03-07-mag-ln-32",]*10

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([1.5,2.4])
    ax.set_xticks(list(range(1,11)))
    #plt.title("Wind to Wind prediction by dataset size")
    plt.xlabel("Training dataset size (# of years)")
    plt.ylabel("Performance (lower is better)")

    # Color and linestyle mapping from legend
    COLORS = {
        "CNN": "#1b9e77",
        "swsCNN": "#d95f02",
        "S3CNN": "#7570b3",
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
            return f"experiments/wind->wind/S3CNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/wind->wind/S3CNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_so3_r_path(ds, run):
        return f"experiments/wind->wind/S3CNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_spins_path(ds, run, rb=False):
        if rb:
            return f"experiments/wind->wind/SWSCNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/wind->wind/SWSCNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_spins_r_path(ds, run):
        return f"experiments/wind->wind/SWSCNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    if PLOT_DATA["CNN_NR_NR"]:
        mean_losses_CNN_NR_NR = [get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_NR, color=COLORS["CNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["CNN_NR_R"]:
        mean_losses_CNN_NR_R = [get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_R, color=COLORS["CNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["CNN_R_R"]:
        mean_losses_CNN_R_R = [get_mean_loss(get_cnn_r_path(ds, CNN_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_R_R, color=COLORS["CNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["SWSCNN_NR_NR"]:
        mean_losses_SWSCNN_NR_NR = [get_mean_loss(get_spins_path(ds, SWSCNN_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_NR_NR, color=COLORS["swsCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SWSCNN_NR_R"]:
        mean_losses_SWSCNN_NR_R = [get_mean_loss(get_spins_path(ds, SWSCNN_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_NR_R, color=COLORS["swsCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SWSCNN_R_R"]:
        mean_losses_SWSCNN_R_R = [get_mean_loss(get_spins_r_path(ds, SWSCNN_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_R_R, color=COLORS["swsCNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["S3CNN_NR_NR"]:
        mean_losses_S3CNN_NR_NR = [get_mean_loss(get_so3_path(ds, S3CNN_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_NR_NR, color=COLORS["S3CNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["S3CNN_NR_R"]:
        mean_losses_S3CNN_NR_R = [get_mean_loss(get_so3_path(ds, S3CNN_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_NR_R, color=COLORS["S3CNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["S3CNN_R_R"]:
        mean_losses_S3CNN_R_R = [get_mean_loss(get_so3_r_path(ds, S3CNN_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_R_R, color=COLORS["S3CNN"], linestyle=LINESTYLES["R/R"])

    plt.legend(
        [
            "CNN NR/NR",
            "CNN NR/R",
            "swsCNN NR/NR",
            "swsCNN NR/R",
            "S3CNN NR/NR",
            "S3CNN NR/R",
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.savefig("figures/windwind_earlyresults_dataset_size.png", bbox_inches='tight')