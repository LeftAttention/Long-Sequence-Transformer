import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

class HourDataset(Dataset):
    
    def __init__(self, root, mode='train', scale=True, x_window=96, y_window=72, inter=48, date_col='date'):
        
        assert mode in ['train', 'test', 'val']
        
        self.root = root
        self.scale = scale
        self.x_window = x_window
        self.y_window = y_window
        self.inter = inter
        self.date_col = date_col
        
        self._read_data()
        
    def _read_data(self):
        le = LabelEncoder()
        scaler = RobustScaler()
        
        df_raw = pd.read_csv(self.root)
        df_data = df_raw[df_raw.columns[1:]]
        
        categorical_cols = list(df_data.select_dtypes(include = 'object').columns)
        
        df_data[categorical_cols] = df_data[categorical_cols].apply(lambda col: le.fit_transform(col))
        data = scaler.fit_transform(df_data.values)
        
        
        df_stamp = df_raw[[self.date_col]]
        df_stamp['date'] = pd.to_datetime(df_stamp.self.date_col)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)

        if self.date_col != 'date':
            data_stamp = df_stamp.drop(['date', self.date_col],1).values
        else:
            data_stamp = df_stamp.drop(['date'],1).values
            
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = torch.from_numpy(self.data_x[s_begin:s_end])
        seq_y = torch.from_numpy(self.data_y[r_begin:r_end])
        seq_x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end])
        seq_y_mark = torch.from_numpy(self.data_stamp[r_begin:r_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
