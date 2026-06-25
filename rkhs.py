import numpy as np


def rkhs_norm_squared(alpha, gram_matrix) -> float:
    """Return alpha' K alpha for a kernel coefficient vector."""
    alpha_np = np.asarray(alpha, dtype=float).ravel()
    gram_np = np.asarray(gram_matrix, dtype=float)
    return float(alpha_np @ gram_np @ alpha_np)
