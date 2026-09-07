import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        w = torch.empty((out_features,in_features))
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(w))
        
        b = torch.empty((out_features))
        self.bias = nn.Parameter(nn.init.zeros_(b))
        

    def forward(self, x):
        return x @ self.weight.T + self.bias
        
