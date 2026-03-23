import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#from matplotlib import cm

import pickle
from sympy import nonlinsolve, total_degree, use
from torch import log_
from tqdm.auto import tqdm
#from tqdm import tqdm
import time

from train_schedule import create_schedule
from run_training import run_training
from run_testing import run_testing

def gpu_task(gpu_id, params):
    """
    Task to handle a single GPU job. This function sanitizes the input parameters and calls the appropriate training or
    testing function.

    Args:
        gpu_id (int): The ID of the GPU to use.
        params (dict): A dictionary containing parameters for the task. These can be task-specific.
    """
    print(f"Process {multiprocessing.current_process().name} is using GPU {gpu_id} with params {params}.")

    ### TRAIN ###
    if params["operation"] == "train":
        run_training(gpu_id=gpu_id, params=params)
        
    ### TEST ###
    elif params["operation"] == "test":
        run_testing(gpu_id=gpu_id, params=params)
        
    else:
        print("Operation is not defined. It should be either \"train\" or \"test\".")
    print(f"Process {multiprocessing.current_process().name} has released GPU {gpu_id}. The proces that ended had params {params}")

def worker(gpu_queue, params):
    """
    Worker function to process tasks using available GPUs.
    """
    gpu_id = gpu_queue.get()  # Lock a GPU
    try:
        gpu_task(gpu_id, params)
    finally:
        gpu_queue.put(gpu_id)  # Release the GPU

if __name__ == "__main__":
    print("Starting job...")

    parser = argparse.ArgumentParser(description='Training job')
    parser.add_argument('--GPU', default="0", type=str, help='Which GPUs to use for training')
    #parser.add_argument('--experiment', default="all", type=str, help='Which experiment to run')

    args = parser.parse_args()

    GPUS = [int(g) for g in args.GPU.split(",")]
    #EXPERIMENT = [g for g in args.experiment.split(",")]

    tasks = create_schedule()

    # Create a queue and populate it with GPU IDs
    gpu_queue = multiprocessing.Queue()
    for gpu_id in GPUS:
        gpu_queue.put(gpu_id)
    
    # Create a pool of workers
    processes = []
    for params in tasks:
        p = multiprocessing.Process(target=worker, args=(gpu_queue, params))
        processes.append(p)
        p.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    

    
    