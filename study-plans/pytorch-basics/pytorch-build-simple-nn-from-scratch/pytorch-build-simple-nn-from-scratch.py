import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features,hidden_size)
        self.w2 = nn.Linear(hidden_size,out_features)
        self.relu =nn.ReLU()

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))
        
