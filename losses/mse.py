from torch import nn


def mse_loss(input, target):
    loss = nn.MSELoss(reduction='mean')
    return loss(input, target)