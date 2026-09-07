import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    return nn.Embedding(vocab_size,d_model,dtype=torch.float32)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    d_model = torch.tensor(d_model,dtype=torch.float32)
    root = torch.sqrt(d_model)
    return embedding(tokens)*root