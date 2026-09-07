import torch

def operation(row):

    return row.max() + torch.log(torch.exp(row-row.max()).sum())

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=torch.float32)
    if method == "mse":
        target = torch.tensor(target, dtype=torch.float32)
        return float(torch.pow(pred-target,2).mean())

    if method == "cross_entropy":
        target = torch.tensor(target,dtype=torch.int32)
        batch_func = torch.vmap(operation)
        target = target.unsqueeze(-1)
        subs  = torch.gather(pred,dim=1,index=target)
        return float((batch_func(pred)-subs).mean())
        
    target = torch.tensor(target, dtype=torch.float32)
    a = torch.abs(pred-target)

    return float(torch.where(a<=delta, (a*a)/2, delta*a -(delta*delta)/2).mean())
