import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == "zeros":
        out = torch.zeros(shape).tolist()
        return out

    if method == "ones":
        out = torch.ones(shape).tolist()
        return out

    out = torch.full(shape,value).tolist()
    return out