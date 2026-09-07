import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)

    if op=="flatten":
        return x.view(-1).tolist()

    if op == "squeeze":
        return x.squeeze().tolist()

    return x.T.tolist()
