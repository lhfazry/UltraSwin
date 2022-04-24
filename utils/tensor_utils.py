import torch
import torch.nn as nn

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = x.mean(dim=1)
        return x