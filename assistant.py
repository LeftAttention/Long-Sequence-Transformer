import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from torch.utils.data import DataLoader
from dataset.dataset import HourDataset
from model.model import LongTimeFormer

from utils.stopping import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

class Assistant(object):
    
    def __init__(self, config):
        
        self.model = LongTimeFormer(
            config.enc_in,
            config.dec_in, 
            config.c_out, 
            config.seq_len, 
            config.label_len,
            config.pred_len, 
            config.factor,
            config.d_model, 
            config.n_heads, 
            config.e_layers,
            config.d_layers, 
            config.d_ff,
            config.dropout, 
            config.attn,
            config.embed,
            config.data[:-1],
            config.activation,
            config.device,
        ).double()
        
        self.criterion =  nn.MSELoss()
        self.model_optim = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
    def to_cuda(self):
        self.model.to(self.device)
        
    def change_mode(self, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()
        
    def validate(self):
        
        total_loss = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(self.val_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()

                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.config.pred_len:,:]).double()
                dec_inp = torch.cat([batch_y[:,:self.config.label_len,:], dec_inp], dim=1).double().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y = batch_y[:,-self.config.pred_len:,:].to(self.device)

                loss = self.criterion(outputs, batch_y) 

                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        return total_loss
