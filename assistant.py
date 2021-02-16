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
        
  
