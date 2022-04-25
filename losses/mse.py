from torch import nn


def mse_loss(input, target):
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    return mse(input, target) + mae(input, target)