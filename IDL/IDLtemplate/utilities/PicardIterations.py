import numpy as np

def picard_iterations(X, D, delta, k_iterations=100):
    for i in range(k_iterations):
        X = np.maximum(0, D@X + delta)
    return X