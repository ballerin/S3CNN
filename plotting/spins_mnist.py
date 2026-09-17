from turtle import color

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pickle

#ours_train_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2025-11-19-lr-ln/epoch_train_losses.pckl", "rb"))
#ours_val_losses = pickle.load(open("experiments/wind->wind/S3CNN/NR/1/2025-11-19-lr-ln/epoch_val_losses.pckl", "rb"))


swscnn_onespin_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SWSCNN/NR/2026-03-07-pc-ln-32-ABLATION-only-one-spin/epoch_train_losses.pckl", "rb"))
swscnn_onespin_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SWSCNN/NR/2026-03-07-pc-ln-32-ABLATION-only-one-spin/epoch_val_losses.pckl", "rb"))

swscnn_bothspins_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/SWSCNN/NR/2025-11-06-pc-ln-32/epoch_train_losses.pckl", "rb"))
swscnn_bothspins_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/SWSCNN/NR/2025-11-06-pc-ln-32/epoch_val_losses.pckl", "rb"))

s3cnn_onespin_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/S3CNN/NR/2026-03-18-lr-ln-32-ABLATION-only-one-spin/epoch_train_losses.pckl", "rb"))
s3cnn_onespin_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/S3CNN/NR/2026-03-18-lr-ln-32-ABLATION-only-one-spin/epoch_val_losses.pckl", "rb"))

s3cnn_bothspins_train_losses = pickle.load(open("experiments/MNIST-scalar-classification/S3CNN/NR/2026-03-13-lr-ln-32/epoch_train_losses.pckl", "rb"))
s3cnn_bothspins_val_losses = pickle.load(open("experiments/MNIST-scalar-classification/S3CNN/NR/2026-03-13-lr-ln-32/epoch_val_losses.pckl", "rb"))

fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (lower is better)")
ax.plot(swscnn_onespin_val_losses, label="swsCNN single spin (val)", color="#D81B60")
ax.plot(swscnn_bothspins_val_losses, label="swsCNN dual spins (val)", color="#1E88E5")
ax.plot(s3cnn_onespin_val_losses, label="S3CNN single spin (val)", color="#FFC107")
ax.plot(s3cnn_bothspins_val_losses, label="S3CNN dual spins (val)", color="#004D40")

trans = blended_transform_factory(ax.transAxes, ax.transData)
ax.plot([-0.01, 0.01], [min(swscnn_onespin_val_losses), min(swscnn_onespin_val_losses)], transform=trans, color="#D81B60", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(swscnn_bothspins_val_losses), min(swscnn_bothspins_val_losses)], transform=trans, color="#1E88E5", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(s3cnn_onespin_val_losses), min(s3cnn_onespin_val_losses)], transform=trans, color="#FFC107", lw=2, clip_on=False)
ax.plot([-0.01, 0.01], [min(s3cnn_bothspins_val_losses), min(s3cnn_bothspins_val_losses)], transform=trans, color="#004D40", lw=2, clip_on=False)

#ax.set_title("Train/val losses comparison for swsCNN single and dual spins in MNIST classification")
#ax.legend(loc="upper center", ncol=2)
ax.set_ylim([0, 0.5])
plt.savefig("figures/one_spin_mnist.png", bbox_inches='tight')