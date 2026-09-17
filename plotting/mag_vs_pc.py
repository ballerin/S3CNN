from turtle import color

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pickle

mag_train_losses = pickle.load(open("experiments/wind->wind/SWSCNN/NR/1/2026-03-07-mag-ln-32/epoch_train_losses.pckl", "rb"))
mag_val_losses = pickle.load(open("experiments/wind->wind/SWSCNN/NR/1/2026-03-07-mag-ln-32/epoch_val_losses.pckl", "rb"))

pc_train_losses = pickle.load(open("experiments/wind->wind/SWSCNN/NR/1/2025-11-11-pc-ln-32/epoch_train_losses.pckl", "rb"))
pc_val_losses = pickle.load(open("experiments/wind->wind/SWSCNN/NR/1/2025-11-11-pc-ln-32/epoch_val_losses.pckl", "rb"))

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (lower is better)")
ax.plot(mag_train_losses, label="Magnitude (train)", color="#1b9e77", linestyle="dotted")
ax.plot(mag_val_losses, label="Magnitude (val)", color="#1b9e77")
ax.plot(pc_train_losses, label="Phase Collapse (train)", color="#d95f02", linestyle="dotted")
ax.plot(pc_val_losses, label="Phase Collapse (val)", color="#d95f02")

trans = blended_transform_factory(ax.transAxes, ax.transData)
ax.plot([-0.01, 0.01], [min(mag_val_losses), min(mag_val_losses)], transform=trans, color="#1b9e77", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(pc_val_losses), min(pc_val_losses)], transform=trans, color="#d95f02", lw=2, clip_on=False)
#ax.set_title("Train/val losses comparison for Magnitude and Phase Collapse")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
ax.set_ylim([0, 10])
plt.savefig("figures/mag_vs_pc_dimension_one.png", bbox_inches='tight')