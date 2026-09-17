from turtle import color

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pickle

unet_train_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2025-11-19-lr-ln/epoch_train_losses.pckl", "rb"))
unet_val_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2025-11-19-lr-ln/epoch_val_losses.pckl", "rb"))

flatshort_train_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2026-02-18-lr-ln-ABLATION-FLAT/epoch_train_losses.pckl", "rb"))
flatshort_val_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2026-02-18-lr-ln-ABLATION-FLAT/epoch_val_losses.pckl", "rb"))

flatlong_train_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2026-02-18-lr-ln-ABLATION-FLAT-LONG/epoch_train_losses.pckl", "rb"))
flatlong_val_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2026-02-18-lr-ln-ABLATION-FLAT-LONG/epoch_val_losses.pckl", "rb"))

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(flatshort_train_losses, label="Train loss")
ax.plot(flatshort_val_losses, label="Validation loss")
ax.set_title("Flat SO(3)-equivarint Network (short network)")
ax.legend(loc="upper right", ncol=1)
ax.set_ylim([0, 10])
plt.savefig("figures/flat_short_comparison_training.png", bbox_inches='tight')

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(flatlong_train_losses, label="Train loss")
ax.plot(flatlong_val_losses, label="Validation loss")
ax.set_title("Flat SO(3)-equivarint Network")
ax.legend(loc="upper right", ncol=1)
ax.set_ylim([0, 10])
plt.savefig("figures/flat_long_comparison_training.png", bbox_inches='tight')

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(unet_train_losses, label="Train loss")
ax.plot(unet_val_losses, label="Validation loss")
ax.set_title("SO(3)-equivarint UNet")
ax.legend(loc="upper right", ncol=1)
ax.set_ylim([0, 10])
plt.savefig("figures/unet_comparison_training.png", bbox_inches='tight')

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (lower is better)")
ax.plot(flatshort_train_losses, label="Shallow Net. (train)", color="#FFBB28", linestyle="dotted")
ax.plot(flatshort_val_losses, label="Shallow Net. (val)", color="#FFBB28")
ax.plot(flatlong_train_losses, label="Deep Net. (train)", color="#000000", linestyle="dotted")
ax.plot(flatlong_val_losses, label="Deep Net. (val)", color="#000000")
ax.plot(unet_train_losses, label="UNet (train)", color="#7570b3", linestyle="dotted")
ax.plot(unet_val_losses, label="UNet (val)", color="#7570b3")

trans = blended_transform_factory(ax.transAxes, ax.transData)
ax.plot([-0.01, 0.01], [min(flatshort_val_losses), min(flatshort_val_losses)], transform=trans, color="#FFBB28", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(flatlong_val_losses), min(flatlong_val_losses)], transform=trans, color="#000000", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(unet_val_losses), min(unet_val_losses)], transform=trans, color="#7570b3", lw=2, clip_on=False)
#ax.set_title("Train/val losses comparison for UNet and flat networks")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.02))
ax.set_ylim([0, 10])
plt.savefig("figures/all_flat_unet_comparison_training.png", bbox_inches='tight')