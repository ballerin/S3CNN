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
    "MAG_NR_NR":    True,
    "MAG_NR_R":     True,
    "MAG_R_R":      False,
    "PC_NR_NR":    True,
    "PC_NR_R":     True,
    "PC_R_R":      False,
}

MAG_NR = ["2026-03-07-mag-ln-32",]*10
MAG_R = ["2026-03-07-mag-ln-32",]*10
PC_NR = ["2025-11-11-pc-ln-32",]*10
PC_R = ["2025-11-11-pc-ln-32",]*10

def sanity_check():
    pass

if __name__ == "__main__":

    sanity_check()
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([1.5,2.4])
    ax.set_xticks(list(range(1,11)))
    #plt.title("Wind to Wind prediction by dataset size")
    plt.xlabel("Training dataset size (# of years)")
    plt.ylabel("Performance (lower is better)")

    # Color and linestyle mapping from legend
    COLORS = {
        "MAG": "#1b9e77",
        "PC": "#d95f02",
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

    if PLOT_DATA["MAG_NR_NR"]:
        mean_losses_MAG_NR_NR = [get_mean_loss(get_spins_path(ds, MAG_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_MAG_NR_NR, color=COLORS["MAG"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["MAG_NR_R"]:
        mean_losses_MAG_NR_R = [get_mean_loss(get_spins_path(ds, MAG_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_MAG_NR_R, color=COLORS["MAG"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["MAG_R_R"]:
        mean_losses_MAG_R_R = [get_mean_loss(get_spins_r_path(ds, MAG_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_MAG_R_R, color=COLORS["MAG"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["PC_NR_NR"]:
        mean_losses_PC_NR_NR = [get_mean_loss(get_spins_path(ds, PC_NR[ds], rb=False)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_PC_NR_NR, color=COLORS["PC"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["PC_NR_R"]:
        mean_losses_PC_NR_R = [get_mean_loss(get_spins_path(ds, PC_NR[ds], rb=True)) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_PC_NR_R, color=COLORS["PC"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["PC_R_R"]:
        mean_losses_PC_R_R = [get_mean_loss(get_spins_r_path(ds, PC_R[ds])) for ds in range(DATASET_SIZES)]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_PC_R_R, color=COLORS["PC"], linestyle=LINESTYLES["R/R"])
    
    plt.legend([
        "Magnitude NR/NR",
        "Magnitude NR/R",
        "Phase Collapse NR/NR",
        "Phase Collapse NR/R",
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))


    plt.savefig("figures/windwind_mag_vs_pc_dataset_size.png", bbox_inches='tight')