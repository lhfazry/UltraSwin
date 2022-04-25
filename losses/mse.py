from torch import nn


def mse_loss(input, target):
    loss = nn.MSELoss(reduction='none')
    return loss(input, target)