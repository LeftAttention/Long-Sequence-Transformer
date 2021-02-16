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

    def train(self):
        if not os.path.exists(self.config.weight_path):
            os.makedirs(self.config.weight_path)
            
        time_now = time.time()
        
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.change_mode('train')
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1
                
                self.model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()
                
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.config.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                self.model_optim.step()

            train_loss = np.average(train_loss)
            val_loss = self.validate()
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss))
            early_stopping(val_loss, self.model, self.config.weight_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.model_optim, epoch+1, self.config.learning_rate)
            
        best_model_path = self.config.weight_path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
