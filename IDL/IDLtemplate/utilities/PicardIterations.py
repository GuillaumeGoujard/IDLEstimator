import numpy as np

def picard_iterations(X, D, delta, k_iterations=100, tol=1e-3):
    i = 0
    while i < k_iterations:
        X_ = np.maximum(0, D@X + delta)
        if np.linalg.norm(X - X_, ord="fro") < tol:
            return X
        X = X_
        i += 1
    return X