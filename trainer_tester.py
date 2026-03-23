import gc
import torch
import torch.nn as nn
import numpy as np

import time
import pickle
from tqdm import tqdm, trange

from modules.SO3Rotator import SO3Rotator

def accuracy(outputs, labels):
    # Get predicted class (index of max logit)
    preds = torch.argmax(outputs, dim=1)
    # Compare predictions to labels and take mean
    return (preds == labels).float().mean()

class Trainer:
    def __init__(
        self,
        gpu_id: int,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        in_datatype: str,
        out_datatype: str,
        optimizer: torch.optim.Optimizer,
        lossfnc: nn.modules.loss._Loss,
        save_every: int,
        fft_method: str,
        logs_path:str,
        autoencoding= False,
        rotate:str = "no",
        share_d_matrices:bool = False,
        save_all_iter_losses: bool = False,
        gradient_accumulation = 0,
        verbose: bool = False,
    ) -> None:
        
        ### Parameters ###
        self.gpu_id = gpu_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.in_datatype = in_datatype
        self.out_datatype = out_datatype
        self.optimizer = optimizer
        self.save_every = save_every
        self.logs_path = logs_path
        self.autoencoding = autoencoding
        self.lossfnc = lossfnc
        self.rotate = rotate
        self.share_d_matrices = share_d_matrices
        self.save_all_iter_losses = save_all_iter_losses
        self.gradient_accumulation = max(gradient_accumulation,1) # At least once per iter
        self.verbose = verbose

        ### Initializations ###
        self.device = torch.device("cuda", self.gpu_id)
        self.iter_train_loss = []
        self.iter_val_loss = []
        self.iter_val_acc = []
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_val_acc = []
        self.epoch_times = []
        self.epochs_with_no_improvements = 0
        self.best_val_loss = -1
        self.best_val_acc = -1
        self.model = self.model.to(self.device)

    def _run_batch(self, source, targets, i = None, train=True, optimizer_step=True, n_acc=1):
        # BEGIN FWD
        time_start = time.time()
        
        if train:
            output = self.model(source)
        else:
            with torch.no_grad():
                output = self.model(source)

        loss = self.lossfnc(output, targets) / n_acc # Scale loss for gradient accumulation
        
        if self.out_datatype == "label":
            acc = accuracy(output, targets)

        # END FWD
        time_fwd = time.time()
        # BEGIN BWD
        
        if train:
            loss.backward()

            if optimizer_step:
                self.optimizer.step()
                self.optimizer.zero_grad() #Zero-grad here for gradient accumulation

        time_bwd = time.time()
        # END BWD

        # Logging
        with open(self.logs_path+"training_log.txt",'a') as ff:
            if train:
                ff.write(f"[TR-{i}] ->  LOSS: {loss.item():.6f}, T_FWD: {time_fwd-time_start:.2f}, T_BWD: {time_bwd-time_fwd:.2f}\n")
                self.iter_train_loss.append(loss.item())
            else:
                if self.out_datatype == "label":
                    ff.write(f"[VAL-{i}] ->  LOSS: {loss.item():.6f}, ACC: {acc.item():.4f}, T_FWD: {time_fwd-time_start:.2f}\n")
                else:
                    ff.write(f"[VAL-{i}] ->  LOSS: {loss.item():.6f}, T_FWD: {time_fwd-time_start:.2f}\n")
                self.iter_val_loss.append(loss.item())
                if self.out_datatype == "label":
                    self.iter_val_acc.append(acc.item())

    def _run_epoch(self, epoch, rotatorX=None, rotatorY=None):
        time_epoch_start = time.time()
        steps = len(self.train_dataloader)
        b_sz = len(next(iter(self.train_dataloader))[0])
        with open(self.logs_path+"training_log.txt",'a') as ff:
            ff.write(f"[GPU-{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {steps}\n")

        # If we do not want to save all the iteration losses, reset the lists and call garbage coll.
        if not self.save_all_iter_losses:
            self.iter_train_loss = []
            self.iter_val_loss = []
            self.iter_val_acc = []
            gc.collect()

        # TRAIN
        # Training pass with rotations if needed
        self.model.train()
        self.optimizer.zero_grad() #Zero-grad now, in case of gradient accumulation

        for i, (source, targets) in enumerate(self.train_dataloader):
            source = source.to(self.device)
            
            if not self.autoencoding: #Save some time if we are autoencoding
                targets = targets.to(self.device)

            if rotatorX != None:
                if self.rotate == "full":
                    source, rotation = rotatorX(source, return_rotation = True)
                elif self.rotate == "beta":
                    source, beta = rotatorX(source, return_beta = True)

            if self.autoencoding:
                targets = source
            else:
                if rotatorY != None:
                    if self.rotate == "full":
                        targets = rotatorY(targets, rotation = rotation)
                    elif self.rotate == "beta":
                        targets = rotatorY(targets, beta = beta)
            
            optimizer_step = False
            n_acc = 1

            if self.gradient_accumulation > 1:
                if (i+1) % self.gradient_accumulation == 0 or (i+1) == len(self.train_dataloader):
                    optimizer_step = True
                if i >= i//self.gradient_accumulation*self.gradient_accumulation:
                    n_acc = len(self.train_dataloader) - i//self.gradient_accumulation*self.gradient_accumulation
                else:
                    n_acc = self.gradient_accumulation
            else:
                optimizer_step = True

            self._run_batch(source, targets, i=i, train=True, optimizer_step=optimizer_step, n_acc=n_acc)

        self.epoch_train_loss.append(np.mean(self.iter_train_loss[-len(self.train_dataloader):]))

        # VALIDATION
        # No rotations for the validation pass (if necessary, this has been precomputed already)
        self.model.eval()
        with torch.no_grad():
            for i, (source, targets) in enumerate(self.val_dataloader):
                source = source.to(self.device)
                if self.autoencoding:
                    targets = source
                else:
                    targets = targets.to(self.device)
                self._run_batch(source, targets, i=i, train=False)
        
        self.epoch_times.append(time.time()-time_epoch_start)

        self.epoch_val_loss.append(np.mean(self.iter_val_loss[-len(self.val_dataloader):]))
        if self.out_datatype == "label":
            self.epoch_val_acc.append(np.mean(self.iter_val_acc[-len(self.val_dataloader):]))
        
        with open(self.logs_path+"training_log.txt",'a') as ff:
            ff.write(f"#### Epoch {epoch} finished with train loss {self.epoch_train_loss[-1]:.6f} and val loss {self.epoch_val_loss[-1]:.6f}\n")
            if self.out_datatype == "label":
                ff.write(f"#### Epoch {epoch} finished with val accuracy {self.epoch_val_acc[-1]:.4f}\n")
            ff.write(f"#### Time elapsed: {time.time()-time_epoch_start:.0f} seconds\n")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = self.logs_path+"snapshots/checkpoint_"+str(epoch)+".pt"
        torch.save(ckp, PATH)

    def train(self, max_epochs: int, early_stopping: int, starting_epoch: int=0):
        if self.rotate != "no":
            # Use the D-matrices stored in the model
            if self.share_d_matrices:
                if self.in_datatype == "complex-vector":
                    rotatorX = SO3Rotator(device=self.device,
                                        rotation_type=self.rotate,
                                        d_matrices_slice=self.model.get_buffered_slice_wigner(n=1),
                                        d_matrices_full=self.model.get_buffered_full_wigner(),
                                        n=1,
                                        n_alpha=self.model.n_alpha,
                                        n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.in_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
                    rotatorX = SO3Rotator(device=self.device,
                                        rotation_type=self.rotate,
                                        d_matrices_slice=self.model.get_buffered_slice_wigner(n=0),
                                        d_matrices_full=self.model.get_buffered_full_wigner(),
                                        n=0,
                                        n_alpha=self.model.n_alpha,
                                        n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.in_datatype == "label":
                    rotatorX = None # This should not happen
                else:
                    raise ValueError(f"Input datatype {self.in_datatype} not supported for rotation {self.rotate}.")
                if self.out_datatype == "complex-vector":
                    rotatorY = SO3Rotator(device=self.device,
                                        rotation_type=self.rotate,
                                        d_matrices_slice=self.model.get_buffered_slice_wigner(n=1),
                                        d_matrices_full=self.model.get_buffered_full_wigner(),
                                        n=1,
                                        n_alpha=self.model.n_alpha,
                                        n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.out_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
                    rotatorY = SO3Rotator(device=self.device,
                                        rotation_type=self.rotate,
                                        d_matrices_slice=self.model.get_buffered_slice_wigner(n=0),
                                        d_matrices_full=self.model.get_buffered_full_wigner(),
                                        n=0,
                                        n_alpha=self.model.n_alpha,
                                        n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.out_datatype == "label":
                    rotatorY = None
                else:
                    raise ValueError(f"Error in rotators init with {self.in_datatype}")
            else:
                if self.in_datatype == "complex-vector":
                    rotatorX = SO3Rotator(device=self.device, rotation_type=self.rotate, n=1,
                                        n_alpha=self.model.n_alpha, n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.in_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
                    rotatorX = SO3Rotator(device=self.device, rotation_type=self.rotate, n=0,
                                        n_alpha=self.model.n_alpha, n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.in_datatype == "label":
                    rotatorX = None # This should not happen
                else:
                    raise ValueError(f"Error in rotators init with {self.in_datatype}")
                if self.out_datatype == "complex-vector":
                    rotatorY = SO3Rotator(device=self.device, rotation_type=self.rotate, n=1,
                                        n_alpha=self.model.n_alpha, n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.out_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
                    rotatorY = SO3Rotator(device=self.device, rotation_type=self.rotate, n=0,
                                        n_alpha=self.model.n_alpha, n_beta=self.model.n_beta,
                                        FFT_L=self.model.n_beta)
                elif self.out_datatype == "label":
                    rotatorY = None
                else:
                    raise ValueError(f"Error in rotators init with {self.out_datatype}")
        else:
            rotatorX = None
            rotatorY = None

        self._save_checkpoint("initial")

        for epoch in trange(starting_epoch, max_epochs, desc="Epochs", disable=not self.verbose):
            self._run_epoch(epoch, rotatorX=rotatorX, rotatorY=rotatorY)
            if self.save_every>0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            last_epoch_data = {
                'train_loss' : self.epoch_train_loss[-1],
                'val_loss' : self.epoch_val_loss[-1],
                'val_acc' : self.epoch_val_acc[-1] if self.out_datatype == "label" else None,
                'time' : self.epoch_times[-1],
            }
            pickle.dump(last_epoch_data, open(self.logs_path+f"training_data/epoch_{epoch}_data.pckl","wb"))

            if self.out_datatype == "label" and self.epoch_val_acc[-1] > self.best_val_acc:
                self.best_val_acc = self.epoch_val_acc[-1]
                self.epochs_with_no_improvements = 0
                self._save_checkpoint("best")
            elif self.out_datatype != "label" and \
                (self.best_val_loss == -1 or self.epoch_val_loss[-1] < self.best_val_loss):
                self.best_val_loss = self.epoch_val_loss[-1]
                self.epochs_with_no_improvements = 0
                self._save_checkpoint("best")
            else:
                self.epochs_with_no_improvements += 1

            if self.epochs_with_no_improvements > early_stopping:
                break
        
        with open(self.logs_path+"training_log.txt",'a') as ff:
            ff.write(f"###################################################\n")
            if self.out_datatype == "label":
                ff.write(f"Best epoch: {np.argmax(self.epoch_val_acc)}\n")
                ff.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
            else:
                ff.write(f"Best epoch: {np.argmin(self.epoch_val_loss)}\n")
                ff.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            ff.write(f"###################################################\n")
        pickle.dump(self.epoch_train_loss, open(self.logs_path+"epoch_train_losses.pckl","wb"))
        pickle.dump(self.epoch_val_loss, open(self.logs_path+"epoch_val_losses.pckl","wb"))


class Tester:
    """
    Class to test a trained model on a test set. The test set is assumed to already be rotated/augmented if necessary.
    Thus no augmentation is performed in this module.

    Depending if the output datatype is "label" or not, accuracy or loss is computed and logged.
    """
    def __init__(
        self,
        model:torch.nn.Module,
        in_datatype:str,
        out_datatype:str,
        dataloader:torch.utils.data.DataLoader,
        lossfnc:nn.modules.loss._Loss,
        gpu_id:int,
        fft_method:str,
        log_path:str,
        losses_path:str,
        train_dataloader:torch.utils.data.DataLoader = None,
        accuracies_path:str = None,
        autoencoding:bool = False,
        rotate:str = "no",
        verbose:bool = False,
    ) -> None:
    
        self.gpu_id = gpu_id
        self.device = torch.device("cuda",self.gpu_id)
        self.model = model.to(self.device)
        self.in_datatype = in_datatype
        self.out_datatype = out_datatype
        self.train_dataloader = train_dataloader
        self.dataloader = dataloader
        self.log_path = log_path
        self.losses_path = losses_path
        self.accuracies_path = accuracies_path
        self.autoencoding = autoencoding
        self.losses = []
        self.accuracies = []
        self.iter_loss = []
        self.lossfnc = lossfnc

    def _run_batch(self, source, targets, i = None, train=False):
        time_start = time.time()

        with torch.no_grad():
            output = self.model(source)

        if self.out_datatype == "label":
            acc = accuracy(output, targets)
        else:
            loss = self.lossfnc(output, targets)

        time_fwd = time.time()
        if train:
            with open(self.log_path,'a') as ff:
                if self.out_datatype == "label":
                    ff.write(f"[PRE-{i}] ->  ACC: {acc.item():.4f}, T_FWD: {time_fwd-time_start:.2f}\n")
                else:
                    ff.write(f"[PRE-{i}] ->  LOSS: {loss.item():.6f}, T_FWD: {time_fwd-time_start:.2f}\n")
        else:
            with open(self.log_path,'a') as ff:
                if self.out_datatype == "label":
                    ff.write(f"[TEST-{i}] ->  ACC: {acc.item():.4f}, T_FWD: {time_fwd-time_start:.2f}\n")
                else:
                    ff.write(f"[TEST-{i}] ->  LOSS: {loss.item():.6f}, T_FWD: {time_fwd-time_start:.2f}\n")
        
        if not train:
            #Append the average loss with a weight corresponding to the size of the batch
            if self.out_datatype == "label":
                self.accuracies.extend([acc.item(),]*source.size(0))
            else:
                self.losses.extend([loss.item(),]*source.size(0))  

    def _run_epoch(self, beta=0):
        b_sz = len(next(iter(self.dataloader))[0])
        with open(self.log_path,'a') as ff:
            ff.write(f"[GPU-{self.gpu_id}] Beta={beta} | Batchsize: {b_sz} | Steps: {len(self.dataloader)}\n")
  
        for i, (source, targets) in enumerate(self.dataloader):
            source = source.to(self.device)
            if self.autoencoding:
                targets = source
            else:
                targets = targets.to(self.device)
            self._run_batch(source, targets, i=i)
    
    def _run_train_epoch(self, backprop=False):
        if self.train_dataloader is None:
            raise ValueError("Train dataloader is None, cannot run training epoch.")
        self.model.train()
        for i, (source, targets) in enumerate(self.train_dataloader):
            source = source.to(self.device)
            
            if not self.autoencoding: #Save some time if we are autoencoding
                targets = targets.to(self.device)

            #if rotatorX != None:
            #    if self.rotate == "full":
            #        source, rotation = rotatorX(source, return_rotation = True)
            #    elif self.rotate == "beta":
            #        source, beta = rotatorX(source, return_beta = True)

            if self.autoencoding:
                targets = source
            #else:
            #    if rotatorY != None:
            #        if self.rotate == "full":
            #            targets = rotatorY(targets, rotation = rotation)
            #        elif self.rotate == "beta":
            #            targets = rotatorY(targets, beta = beta)

            self._run_batch(source, targets, i=i, train=True)
        self.model.eval()
        
    def test(self, run_train_epoch:bool=False):
        if run_train_epoch:
            self._run_train_epoch(backprop=False)
            
        # Set the model to evaluation mode (no dropout, batch normalization in eval mode, etc.)
        self.model.eval()

        self._run_epoch()

        with open(self.log_path,'a') as ff:
            if self.out_datatype == "label":
                ff.write(f"Average accuracy: {np.mean(self.accuracies):.4f}\n")
            else:
                ff.write(f"Maximum loss: {np.max(self.losses):.6f}\n")
                ff.write(f"Minimum loss: {np.min(self.losses):.6f}\n")
                ff.write(f"Average loss: {np.mean(self.losses):.6f}\n")
        if self.out_datatype == "label":
            pickle.dump(self.accuracies, open(self.accuracies_path,"wb"))
        else:
            pickle.dump(self.losses, open(self.losses_path,"wb"))