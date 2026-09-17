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
    "CNN_R_R":      True,
    "S3CNN_NR_NR":    True,
    "S3CNN_NR_R":     True,
    "S3CNN_R_R":      True,
    "SWSCNN_NR_NR":    True,
    "SWSCNN_NR_R":     True,
    "SWSCNN_R_R":      True,
}

CNN_NR = ["old_run",]*10

CNN_R = ["2026-03-02-run1",]*10

S3CNN_NR = ["2026-02-17",]*10

S3CNN_R = ["2026-02-17",]*10

SWSCNN_NR = ["2026-02-19-pc-ln-16",]*10

SWSCNN_R = ["2026-02-19-pc-ln-16",]*10

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([1.5,2.8])
    ax.set_xticks(list(range(1,11)))
    #plt.title("Wind autoencoder by dataset size")
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

    def get_mean_loss(file_or_list):
        """
        Given a filename or a list of filenames, load and average the mean losses.
        """
        if isinstance(file_or_list, list):
            # Recursively handle nested lists
            means = [get_mean_loss(f) for f in file_or_list]
            return np.mean(means)
        else:
            # file_or_list is a string (filename)
            return np.mean(pickle.load(open(file_or_list, "rb")))

    def build_path(base, ds, run, test_type):
        return f"experiments/autoencoder/{base}/{test_type}/{ds+1}/{run}/{test_type}_test_losses.pckl"

    # Helper to build the correct filename for CNN and S3CNN (with NR/R distinction)
    def get_cnn_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/CNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/CNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_cnn_r_path(ds, run):
        return f"experiments/autoencoder/CNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_so3_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/S3CNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/S3CNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_so3_r_path(ds, run):
        return f"experiments/autoencoder/S3CNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_spins_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/SWSCNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/SWSCNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_spins_r_path(ds, run):
        return f"experiments/autoencoder/SWSCNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    if PLOT_DATA["CNN_NR_NR"]:
        mean_losses_CNN_NR_NR = [
            get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=False)) if not isinstance(CNN_NR[ds], list)
            else get_mean_loss([get_cnn_path(ds, run, rb=False) for run in CNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_NR, color=COLORS["CNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["CNN_NR_R"]:
        mean_losses_CNN_NR_R = [
            get_mean_loss(get_cnn_path(ds, CNN_NR[ds], rb=True)) if not isinstance(CNN_NR[ds], list)
            else get_mean_loss([get_cnn_path(ds, run, rb=True) for run in CNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_NR_R, color=COLORS["CNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["CNN_R_R"]:
        mean_losses_CNN_R_R = [
            get_mean_loss(get_cnn_r_path(ds, CNN_R[ds])) if not isinstance(CNN_R[ds], list)
            else get_mean_loss([get_cnn_r_path(ds, run) for run in CNN_R[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_CNN_R_R, color=COLORS["CNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["S3CNN_NR_NR"]:
        mean_losses_S3CNN_NR_NR = [
            get_mean_loss(get_so3_path(ds, S3CNN_NR[ds], rb=False)) if not isinstance(S3CNN_NR[ds], list)
            else get_mean_loss([get_so3_path(ds, run, rb=False) for run in S3CNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_NR_NR, color=COLORS["S3CNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["S3CNN_NR_R"]:
        mean_losses_S3CNN_NR_R = [
            get_mean_loss(get_so3_path(ds, S3CNN_NR[ds], rb=True)) if not isinstance(S3CNN_NR[ds], list)
            else get_mean_loss([get_so3_path(ds, run, rb=True) for run in S3CNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_NR_R, color=COLORS["S3CNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["S3CNN_R_R"]:
        mean_losses_S3CNN_R_R = [
            get_mean_loss(get_so3_r_path(ds, S3CNN_R[ds])) if not isinstance(S3CNN_R[ds], list)
            else get_mean_loss([get_so3_r_path(ds, run) for run in S3CNN_R[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_S3CNN_R_R, color=COLORS["S3CNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["SWSCNN_NR_NR"]:
        mean_losses_SWSCNN_NR_NR = [
            get_mean_loss(get_spins_path(ds, SWSCNN_NR[ds], rb=False)) if not isinstance(SWSCNN_NR[ds], list)
            else get_mean_loss([get_spins_path(ds, run, rb=False) for run in SWSCNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_NR_NR, color=COLORS["swsCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SWSCNN_NR_R"]:
        mean_losses_SWSCNN_NR_R = [
            get_mean_loss(get_spins_path(ds, SWSCNN_NR[ds], rb=True)) if not isinstance(SWSCNN_NR[ds], list)
            else get_mean_loss([get_spins_path(ds, run, rb=True) for run in SWSCNN_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_NR_R, color=COLORS["swsCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SWSCNN_R_R"]:
        mean_losses_SWSCNN_R_R = [
            get_mean_loss(get_spins_r_path(ds, SWSCNN_R[ds])) if not isinstance(SWSCNN_R[ds], list)
            else get_mean_loss([get_spins_r_path(ds, run) for run in SWSCNN_R[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SWSCNN_R_R, color=COLORS["swsCNN"], linestyle=LINESTYLES["R/R"])

    plt.savefig("figures/autoencoder_dataset_size.png", bbox_inches='tight')