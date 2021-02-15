import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

class Days_dataset(Dataset):
    
    def __init__(self, root, mode='train', scale=True, x_window=96, y_window=72, inter=48):
        
        assert mode in ['train', 'test', 'val']
        
        self.root = root
        self.scale = scale
        self.x_window = x_window
        self.y_window = y_window
        self.inter = inter
        
        self._read_data()
        
