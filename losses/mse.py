from torch import nn
import torchmetrics


def mse_loss(input, target):
    mse = nn.MSELoss()
    #mae = nn.L1Loss() #torchmetrics.MeanAbsoluteError()
    #return mse(input, target) + mae(input, target)
    #loss = nn.HuberLoss()
    return mse(input, target)