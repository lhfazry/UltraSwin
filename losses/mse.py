from torch import nn
import torchmetrics


def mse_loss(input, target):
    #mse = nn.MSELoss()
    mae = torchmetrics.MeanAbsoluteError()
    #return mse(input, target) + mae(input, target)
    return mae(input, target)