from torch import nn
import torch
import torchmetrics

class RMSE(torchmetrics.Metric):
    def __init__(self):
        self.add_state("sum_squared_errors", default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state("n_observations", default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)