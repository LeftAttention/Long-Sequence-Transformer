import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, learning_rate):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    
    lr_adjust = learning_rate * (0.5 ** ((epoch-1) // 1))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_adjust
    print('Updating learning rate to {}'.format(lr_adjust))
