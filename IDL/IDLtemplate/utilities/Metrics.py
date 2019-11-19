import numpy as np


def fenchtel_div(X,Y):
    Z = 0.5*(np.multiply(X, X) + np.multiply(np.maximum(0, Y), np.maximum(0, Y))) - np.multiply(X, Y)
    return np.mean(Z, axis=1)

def loss(y, U, theta, X):
    test_satisfies_contraints(y, U, theta, X)
    lambda_ = None
    L2Loss(y, U, theta, X) + lambda_@fenchtel_div(X, )


def test_satisfies_contraints(y, U, theta, X):
    pass


def L2Loss(y, U, theta, X):
    pass

