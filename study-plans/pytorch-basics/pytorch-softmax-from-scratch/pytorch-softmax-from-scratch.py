import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max = logits.max(dim=1,keepdim=True).values
    logits = logits-max
    logits = torch.exp(logits)
    sum = logits.sum(dim=1,keepdim=True)
    logits = logits/sum
    return logits
