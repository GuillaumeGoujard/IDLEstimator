from utilities import Metrics
import numpy as np


def update_dual(U, theta, X, alpha=0.1, epsilon=0.01, k=1):
    A, B, c, D, E, f, Lambda, m = theta["A"], theta["B"], theta["c"], theta["D"],theta["E"], theta["f"],theta["Lambda"], theta["m"]
    F = Metrics.fenchtel_div(X, D@X + E@U + f.reshape((f.shape[0],1))@np.ones((1,m)))
    indicator = np.zeros(F.shape)
    indicator[F > epsilon] = 1
    dlambda = alpha*np.multiply(F, indicator)
    return np.diag(Lambda) + dlambda


def update_dual2(lambda_dual):
    #TODO
    pass

def update_dual3(last_multiplier, U, theta, X, alpha=0.1, epsilon=0.01, k=1):
    A, B, c, D, E, f, Lambda, m = theta["A"], theta["B"], theta["c"], theta["D"],theta["E"], theta["f"],theta["Lambda"], theta["m"]
    F = Metrics.fenchtel_div(X, D@X + E@U + f.reshape((f.shape[0],1))@np.ones((1,m)))
    indicator = np.zeros(F.shape)
    indicator[F > epsilon] = 1
    if k == 0:
        c_k = 1
    else:
        alpha_k = 1 - 1 / (2 * np.sqrt(k))
        print("k = ", alpha_k)
        c_k = alpha_k*(last_multiplier)/np.linalg.norm(np.multiply(F, indicator))
    print("c_k used is ", c_k)
    dlambda = c_k*np.multiply(F, indicator)
    print("norm of dlambda = ", np.linalg.norm(dlambda))
    return np.diag(Lambda) + dlambda, np.linalg.norm(dlambda)
