import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    x = torch.tensor(values,dtype =torch.float32,requires_grad=True)
    y = torch.add(torch.pow(x,3),torch.mul(x,2)).sum()
    y.backward()
    return x.grad.tolist()
