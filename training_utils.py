import matplotlib.pyplot as plt
from trainer_tester import Trainer

def plot_data(trainer: Trainer, path, min_loss=0, max_loss=100):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(trainer.epoch_train_loss, label="Train loss")
    ax.plot(trainer.epoch_val_loss, label="Validation loss")
    ax.set_title("Train and validation losses by epochs")
    ax.set_ylim([min_loss, max_loss])
    plt.savefig(path + "epoch_losses.png")

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Train  Loss")
    ax.plot(trainer.iter_train_loss)
    plt.savefig(path + "train_losses.png")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Validation  Loss")
    ax.plot(trainer.iter_val_loss)
    plt.savefig(path + "val_losses.png")
    """