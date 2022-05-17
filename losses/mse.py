from torch import nn
import torchmetrics
import torch.nn.functional as F


def mse_loss(input, target):
    mse = nn.MSELoss()
    
    #mae = nn.L1Loss() #torchmetrics.MeanAbsoluteError()
    #return mse(input, target) + mae(input, target)
    hubber_loss = nn.HuberLoss()
    return mse(input, target)