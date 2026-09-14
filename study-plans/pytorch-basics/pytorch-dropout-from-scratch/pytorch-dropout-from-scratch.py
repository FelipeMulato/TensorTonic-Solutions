import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if(not self.training):
            return x
        shape = x.size()
        if(self.p==1.0):
            return torch.zeros((shape))

        m = torch.rand((shape),dtype=torch.float32)
        result= torch.where(m<self.p,0,x)
        result = (1/(1-self.p))*result
        return result
