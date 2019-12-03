import numpy as np


def fenchtel_div(X, Y):
    Z = 0.5*(np.multiply(X, X) + np.multiply(np.maximum(0, Y), np.maximum(0, Y))) - np.multiply(X, Y)
    return np.mean(Z, axis=1)


def loss(U, y, theta, X):
    test_satisfies_contraints(y, U, theta, X)
    lambda_ = np.diag(theta["Lambda"])
    m = theta["m"]
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    y_fenchtel = D@X + E@U + f@np.ones((1,m))
    return L2Loss(U, y, theta, X) + lambda_@fenchtel_div(X, y_fenchtel)


def test_satisfies_contraints(y, U, theta, X):
    pass


def L2Loss(U, y, theta, X):
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    m = theta["m"]
    M = A@X + B@U + c@np.ones((1,m)) - y
    return (1/(2*m))*np.linalg.norm(M, ord="fro")**2


def fenchtel_error(theta, X, U, lambda_=None):
    m = theta["m"]
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    y_fenchtel = D @ X + E @ U + f @ np.ones((1, m))
    if lambda_ is None:
        lambda_ = np.diag(theta["Lambda"])
    return np.float(lambda_ @ fenchtel_div(X, y_fenchtel))



