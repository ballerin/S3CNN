"""
This is a an example template for scheduling training/testing jobs.

Create a new file named `train_schedule.py` in the same directory as this template, with the jobs
you wish to run.
"""

def create_schedule():
    
    schedule = [
        {
            "operation": "train",
            "experiment": "MNIST-scalar-classification",
            "rotate":"no",
            "model_type":"CNNMNIST",
            "batch_size": 512,
            "total_epochs": 1000,
            "early_stopping": 50,
            "save_every": -1,
            "log_id": "21-08-2025",
        },
        {
            "operation": "train",
            "experiment": "MNIST-scalar-classification",
            "rotate":"beta",
            "model_type":"SO3MNIST",
            "batch_size": 1,
            "total_epochs": 100,
            "early_stopping": 10,
            "save_every": 1,
            "log_id": "21-08-2025",
            "starting_epoch": 4, # The first epoch is #4
            "pretrained_wights": "experiments/MNIST-scalar-classification/SO3/NR/23-08-2025/snapshots/checkpoint_best.pt"
        },
        {
            "operation": "train",
            "experiment": "temp->wind",
            "dataset_size": 1,
            "rotate": "no",
            "model_type": "SpinsSO3UNet",
            "nonlinearity": "magnitude",
            "batch_size": 32,
            "batch_norm": False,
            "total_epochs": 1000,
            "early_stopping": 10,
            "save_every": -1,
            "log_id": "no_batch_32_magnitude",
        },
    ]

    return schedule