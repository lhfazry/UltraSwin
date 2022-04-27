from torch import nn
import torchmetrics


def mse_loss(input, target):
    mse = nn.MSELoss()
    #mae = nn.L1Loss() #torchmetrics.MeanAbsoluteError()
    #return mse(input, target) + mae(input, target)
    hubber_loss = nn.HuberLoss()
    return hubber_loss(input, target)