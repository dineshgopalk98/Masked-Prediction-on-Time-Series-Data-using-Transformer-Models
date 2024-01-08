import torch
from loss import *
from collections import OrderedDict
import sys
import builtins
import numpy as np


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console = True):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader,1):
            
            X, targets, target_masks, padding_masks = batch   #, IDs
            targets = targets.to(self.device)
            #print(targets)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            #print(target_masks)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore-
            #print(padding_masks)

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss.view(-1))  # mean loss (over active elements) used for optimization
            print(loss)

            if self.l2_reg:
                total_loss = loss + self.l2_reg * l2_reg_loss(self.model)#mean_loss
            else:
                total_loss = loss#mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            # if i % self.print_interval == 0:
            ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            self.print_callback(i, metrics, prefix='Training ' + ending)

            #with torch.no_grad():
            total_active_elements += len(loss.view(-1))
            epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        
        return epoch_num, epoch_loss

    def evaluate(self, epoch_num=None, keep_all=True):
        
        self.model = self.model.eval()
        #epoch_loss = 0  # total loss of epoch
#         total_active_elements = 0  # total unmasked elements in epoch
#         for i, batch in enumerate(self.dataloader,1):
            
#             X, targets, target_masks, padding_masks = batch   #, IDs
#             targets = targets.to(self.device)
#             #print(targets)
#             target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
#             #print(target_masks)
#             padding_masks = padding_masks.to(self.device)  # 0s: ignore-
#             #print(padding_masks)

#             predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

#             # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
#             target_masks = target_masks * padding_masks.unsqueeze(-1)
#             loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
#             batch_loss = torch.sum(loss)
#             mean_loss = batch_loss / len(loss.view(-1))  # mean loss (over active elements) used for optimization
#             print(loss)

#             if self.l2_reg:
#                 total_loss = loss + self.l2_reg * l2_reg_loss(self.model)#mean_loss
#             else:
#                 total_loss = loss#mean_loss

#             # Zero gradients, perform a backward pass, and update the weights.
#             self.optimizer.zero_grad()
#             total_loss.backward()

#             # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
#             self.optimizer.step()

#             metrics = {"loss": mean_loss.item()}
#             #if i % self.print_interval == 0:
#             ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
#             self.print_callback(i, metrics, prefix='Evaluating ' + ending)

#             with torch.no_grad():
#                 total_active_elements += len(loss.view(-1))
#                 epoch_loss += batch_loss.item()  # add total loss of batch

#         epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
#         self.epoch_metrics['epoch'] = epoch_num
#         self.epoch_metrics['loss'] = epoch_loss
        
#         return self.epoch_metrics
        epoch_loss = 0  
        total_active_elements = 0  

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader,1):

            X, targets, target_masks, padding_masks = batch  
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  
            padding_masks = padding_masks.to(self.device)  

           

            predictions = self.model(X.to(self.device), padding_masks)  

           
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss.view(-1))  

            if keep_all:
                per_batch['target_masks'].append(target_masks.cpu())
                per_batch['targets'].append(targets.cpu())
                per_batch['predictions'].append(predictions.cpu())
                per_batch['metrics'].append([loss.cpu()])
                

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)
            
            #with torch.no_grad():
            total_active_elements += len(loss.view(1))
            epoch_loss += batch_loss  

        epoch_loss = epoch_loss / total_active_elements  
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics