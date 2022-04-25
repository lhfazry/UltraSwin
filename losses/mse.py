from torch import nn


def mse_loss(input, target):
    loss = nn.MSELoss()
    return loss(input, target)