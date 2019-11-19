from utilities import Metrics
import numpy as np


def update_dual(theta, X, U, alpha=0.1, epsilon=0.1):
    A, B, c, D, E, f, Lambda = theta
    m = X.shape[1]
    F = Metrics.fenchtel_div(X, D@X + E@U + f.reshape((f.shape[0],1))@np.ones((1,m)))
    indicator = np.zeros(F.shape)
    indicator[F > epsilon] = 1
    dlambda = alpha*np.multiply(F, indicator)
    return np.diag(Lambda) + dlambda


def update_dual2(lambda_dual):
    #TODO
    pass