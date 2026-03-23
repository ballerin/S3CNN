from turtle import color

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pickle

#ours_train_losses = pickle.load(open("experiments/wind->wind/SO3/NR/1/2025-11-19-lr-ln/epoch_train_losses.pckl", "rb"))
#ours_val_losses = pickle.load(open("experiments/wind->wind/SO3/NR/1/2025-11-19-lr-ln/epoch_val_losses.pckl", "rb"))


scnn_onespin_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SPINS/NR/2026-03-07-pc-ln-32-ABLATION-only-one-spin/epoch_train_losses.pckl", "rb"))
scnn_onespin_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SPINS/NR/2026-03-07-pc-ln-32-ABLATION-only-one-spin/epoch_val_losses.pckl", "rb"))

scnn_bothspins_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SPINS/NR/2025-11-06-pc-ln-32/epoch_train_losses.pckl", "rb"))
scnn_bothspins_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SPINS/NR/2025-11-06-pc-ln-32/epoch_val_losses.pckl", "rb"))

gscnn_onespin_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SO3/NR/2026-03-18-lr-ln-32-ABLATION-only-one-spin/epoch_train_losses.pckl", "rb"))
gscnn_onespin_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SO3/NR/2026-03-18-lr-ln-32-ABLATION-only-one-spin/epoch_val_losses.pckl", "rb"))

gscnn_bothspins_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SO3/NR/2026-03-13-lr-ln-32/epoch_train_losses.pckl", "rb"))
gscnn_bothspins_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SO3/NR/2026-03-13-lr-ln-32/epoch_val_losses.pckl", "rb"))

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (lower is better)")
ax.plot(scnn_onespin_val_losses, label="sCNN single spin (val)", color="#FFBB28")
ax.plot(scnn_bothspins_val_losses, label="sCNN dual spins (val)", color="#2ca02c")
ax.plot(gscnn_onespin_val_losses, label="gCNN single spin (val)", color="#7570b3")
ax.plot(gscnn_bothspins_val_losses, label="gCNN dual spins (val)", color="#d62728")

trans = blended_transform_factory(ax.transAxes, ax.transData)
ax.plot([-0.01, 0.01], [min(scnn_onespin_val_losses), min(scnn_onespin_val_losses)], transform=trans, color="#FFBB28", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(scnn_bothspins_val_losses), min(scnn_bothspins_val_losses)], transform=trans, color="#2ca02c", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(gscnn_onespin_val_losses), min(gscnn_onespin_val_losses)], transform=trans, color="#7570b3", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(gscnn_bothspins_val_losses), min(gscnn_bothspins_val_losses)], transform=trans, color="#d62728", lw=2, clip_on=False)

#ax.set_title("Train/val losses comparison for sCNN single and dual spins in MNIST classification")
#ax.legend(loc="upper center", ncol=2)
ax.set_ylim([0, 0.5])
plt.savefig("figures/one_spin_mnist.png")