import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    vetor = np.asarray(x)
    relu = np.asarray(np.maximum(0,vetor))
    return relu