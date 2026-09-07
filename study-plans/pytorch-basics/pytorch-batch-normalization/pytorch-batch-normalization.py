import torch
def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    media = X.mean(dim=0, keepdim=True)

    variancia = X.var(dim=0, unbiased=False, keepdim=True)
    
    X_hat = (X-media) / torch.sqrt(variancia + eps)
    
    return X_hat *gamma+beta
