import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x=torch.tensor(x,dtype=torch.float32,requires_grad=True)
    if method == "relu":
        out = torch.where(x>0,x, torch.mul(x,0))
        return out.tolist()
    if method == "sigmoid":
        return torch.div(1,torch.add(1,torch.exp(torch.mul(x,-1)))).tolist()

    if method == 'tanh':
        plus = torch.exp(x) 
        minus =torch.exp(torch.mul(x,-1))
        return torch.div(torch.sub(plus,minus),torch.add(plus,minus) ).tolist()

    out = torch.where(x>0,x, torch.mul(x,0.01))
    return out.tolist()

        