import numpy as np
import torch
from utils.Callbacks import CallBack
from time import time

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Trainer class to train and validate a model from scratch or resume from checkpoint
    """
    def __init__(self, model, opt, lr, l2, loss_fn, callbacks : dict, device='cuda', filepath='.', model_checkpoint=None):
        """
        Args:
            model: model to train
            opt: optimizer to train model parameters
            lr: learning rate for optimizer
            l2: L2 penalty for weights
            loss_fn: loss function for training
            callbacks: callback params : dict {'model_checkpoint_interval', 'early_stop_patience', 'early_stop_min_delta'}
            device: device for compuations ('cpu'/'cuda')
            filepath: filepath for storing logs, checkpoints, best_model
            model_checkpoint: path to resume training from given checkpoint
        """
        self.device = device
        if not torch.cuda.is_available(): self.device='cpu'

        self.model = model.to(self.device)
        self.opt = opt(self.model.parameters(), lr=lr, weight_decay=l2)

        if model_checkpoint is not None:
            state_dict = torch.load(f'{model_checkpoint}/best_model/model.pt')
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])
        
        self.loss_fn = loss_fn
        self.filepath = filepath
        self.callbacks = callbacks

    def train_one_epoch(self, dl, writer, epoch, global_step):
        """ training one epoch """
        self.model.train()
        total_loss = 0
        hits = 0
        total_samples = 0
        
        for batch in dl:
            x, y, mask = [ele.to(self.device) for ele in batch]
            
            out = self.model(x, mask)
            loss = self.loss_fn(out, y)
            
            #writer
            writer.add_scalar("train_step_loss", loss.item(), global_step=global_step)
            global_step += 1
    
            #training
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
    
            #logging
            total_loss += loss.item()*x.size(0)
            total_samples += x.size(0)
            hits += ((out>=0.5)==y).sum().item()

        total_loss/=total_samples
        accuracy = hits/total_samples
        writer.add_scalar("train_loss", total_loss, global_step = epoch)
        writer.add_scalar("train_acc", accuracy, global_step = epoch)
        return total_loss, accuracy, global_step

    def validation(self, dl, writer, epoch, global_step):
        """ validation after training one epoch """
        self.model.eval()
        total_loss = 0
        hits = 0
        total_samples = 0
        for batch in dl:
            with torch.no_grad():
                x, y, mask = [ele.to(self.device) for ele in batch]
                
                out = self.model(x, mask)
                loss = self.loss_fn(out, y)
                writer.add_scalar("val_step_loss", loss.item(), global_step = global_step)
                global_step +=1
    
            #logging
            hits += ((out>=0.5)==y).sum().item()
            total_loss += loss.item()*x.size(0)
            total_samples += x.size(0)
    
        
        total_loss/=total_samples
        accuracy = hits/total_samples
        
        writer.add_scalar("val_loss", total_loss, global_step = epoch)
        writer.add_scalar("val_acc", accuracy, global_step = epoch)
    
        return total_loss, accuracy, global_step

    def train(self, epochs, train_dl, val_dl):
        """train function

        Args:
            epochs: max number of epochs to train model on
            train_dl: training dataloader
            val_dl: validation dataloader
        """
        filename_suffix = self.filepath.split("/")[-1]
        tensorboard_dir = "/home/shashank/repos/tensorboard_runs/"
        writer = SummaryWriter(tensorboard_dir, filename_suffix=filename_suffix)
        callback = CallBack(filepath=self.filepath, callbacks=self.callbacks, model=self.model)
        train_global_step = 0
        val_global_step = 0
        for epoch in range(epochs):
            ep_start = time()
            print(f'Epoch {epoch+1}:')
            train_loss, train_acc, train_global_step = self.train_one_epoch(train_dl, writer, epoch, train_global_step)
            print(f'Training Loss: {np.round(train_loss, 2)} || Training Accuracy: {np.round(train_acc, 2)}')
            val_loss, val_acc, val_global_step = self.validation(val_dl, writer, epoch, val_global_step)
            print(f'Validation Loss: {np.round(val_loss, 2)} || Validation Accuracy: {np.round(val_acc, 2)}')
            ep_end = time()
            print(f'Time: {np.round(ep_end-ep_start, 0)}\n', flush=True)
            
            if callback(self.model.state_dict(), self.opt.state_dict(), train_loss, train_acc, val_loss, val_acc): break

        self.model.load_state_dict(torch.load(f'{self.filepath}/best_model/model.pt')['model_state_dict'])
        torch.save(self.model, f'{self.filepath}/best_model/model.bin')
    
        return















