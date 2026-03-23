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
    "SO3_NR_NR":    True,
    "SO3_NR_R":     True,
    "SO3_R_R":      True,
    "SPINS_NR_NR":    True,
    "SPINS_NR_R":     True,
    "SPINS_R_R":      True,
}

CNN_NR = ["old_run",]*10

CNN_R = ["2026-03-02-run1",]*10

SO3_NR = ["2026-02-17",]*10

SO3_R = ["2026-02-17",]*10

SPINS_NR = ["2026-02-19-pc-ln-16",]*10

SPINS_R = ["2026-02-19-pc-ln-16",]*10

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([1.5,2.8])
    ax.set_xticks(list(range(1,11)))
    #plt.title("Wind autoencoder by dataset size")
    plt.xlabel("Training dataset size (# of years)")
    plt.ylabel("Performance (lower is better)")

    # Color and linestyle mapping from legend
    COLORS = {
        "CNN": "#1b9e77",
        "sCNN": "#d95f02",
        "FsCNN": "#7570b3",
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

    # Helper to build the correct filename for CNN and SO3 (with NR/R distinction)
    def get_cnn_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/CNN/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/CNN/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_cnn_r_path(ds, run):
        return f"experiments/autoencoder/CNN/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_so3_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/SO3/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/SO3/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_so3_r_path(ds, run):
        return f"experiments/autoencoder/SO3/RB/{ds+1}/{run}/RB_test_losses.pckl"

    def get_spins_path(ds, run, rb=False):
        if rb:
            return f"experiments/autoencoder/SPINS/NR/{ds+1}/{run}/RB_test_losses.pckl"
        else:
            return f"experiments/autoencoder/SPINS/NR/{ds+1}/{run}/NR_test_losses.pckl"

    def get_spins_r_path(ds, run):
        return f"experiments/autoencoder/SPINS/RB/{ds+1}/{run}/RB_test_losses.pckl"

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
    if PLOT_DATA["SO3_NR_NR"]:
        mean_losses_SO3_NR_NR = [
            get_mean_loss(get_so3_path(ds, SO3_NR[ds], rb=False)) if not isinstance(SO3_NR[ds], list)
            else get_mean_loss([get_so3_path(ds, run, rb=False) for run in SO3_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_NR_NR, color=COLORS["FsCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SO3_NR_R"]:
        mean_losses_SO3_NR_R = [
            get_mean_loss(get_so3_path(ds, SO3_NR[ds], rb=True)) if not isinstance(SO3_NR[ds], list)
            else get_mean_loss([get_so3_path(ds, run, rb=True) for run in SO3_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_NR_R, color=COLORS["FsCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SO3_R_R"]:
        mean_losses_SO3_R_R = [
            get_mean_loss(get_so3_r_path(ds, SO3_R[ds])) if not isinstance(SO3_R[ds], list)
            else get_mean_loss([get_so3_r_path(ds, run) for run in SO3_R[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SO3_R_R, color=COLORS["FsCNN"], linestyle=LINESTYLES["R/R"])
    if PLOT_DATA["SPINS_NR_NR"]:
        mean_losses_SPINS_NR_NR = [
            get_mean_loss(get_spins_path(ds, SPINS_NR[ds], rb=False)) if not isinstance(SPINS_NR[ds], list)
            else get_mean_loss([get_spins_path(ds, run, rb=False) for run in SPINS_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_NR_NR, color=COLORS["sCNN"], linestyle=LINESTYLES["NR/NR"])
    if PLOT_DATA["SPINS_NR_R"]:
        mean_losses_SPINS_NR_R = [
            get_mean_loss(get_spins_path(ds, SPINS_NR[ds], rb=True)) if not isinstance(SPINS_NR[ds], list)
            else get_mean_loss([get_spins_path(ds, run, rb=True) for run in SPINS_NR[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_NR_R, color=COLORS["sCNN"], linestyle=LINESTYLES["NR/R"])
    if PLOT_DATA["SPINS_R_R"]:
        mean_losses_SPINS_R_R = [
            get_mean_loss(get_spins_r_path(ds, SPINS_R[ds])) if not isinstance(SPINS_R[ds], list)
            else get_mean_loss([get_spins_r_path(ds, run) for run in SPINS_R[ds]])
            for ds in range(DATASET_SIZES)
        ]
        ax.plot(range(1,DATASET_SIZES+1), mean_losses_SPINS_R_R, color=COLORS["sCNN"], linestyle=LINESTYLES["R/R"])

    plt.savefig("figures/autoencoder_dataset_size.png")