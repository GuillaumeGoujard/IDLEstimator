import numpy as np

def alpha_theta(X_, U):
    #TODO
    pass

def gradient_descent_theta(theta, X, U, Y):
    """
    Returns the gradient of theta
    :param theta: a dictionary
    :param X: hidden variables
    :param U: input data
    :param Y: output data
    :return: grad_theta: dictionary containing gradients of elemnts in theta
    """
    grad_theta = {}

    m = np.shape(theta["c"])
    AB_Omega = Omega_ABc(theta, X=X, U=U, Y=Y)
    grad_theta["A"] = AB_Omega @ X.T
    grad_theta["B"] = AB_Omega @ U.T
    grad_theta["c"] = AB_Omega @ np.ones(m)
    xnorm = np.linalg.norm(X, ord="fro")
    unorm = np.linalg.norm(U, ord="fro")
    lip_theta_one = (1/m) * max(m, xnorm ** 2, unorm ** 2, np.linalg.norm(X @ U.T, ord="fro"))

    DEF_Omega = Omega_DEfLambda(m, D=theta["D"], E=theta["E"], F=theta["F"],
                                Lambda=theta["Lambda"], X=X, U=theta["U"])

    grad_theta["D" ] = DEF_Omega @ X.T
    grad_theta["E"] = DEF_Omega @ U.T
    grad_theta["f"] = DEF_Omega @ np.ones(m)

    lip_theta_two = (np.max(theta["Lambda"]) / m) * max(m, xnorm ** 2, unorm ** 2, xnorm, unorm)

    return grad_theta, (lip_theta_one, lip_theta_two)

def alpha_x(theta):
    #TODO
    pass

def gradient_descent_x(theta, X, U, y):
    m = np.shape(theta["c"])
    A = theta["A"]
    B = theta["B"]
    D = theta["D"]
    E = theta["E"]
    f = theta["f"]
    c = theta["c"]
    Lambda = theta["Lambda"]
    # Make sure that the shapes from the outer products c @ np.ones(m) etc. are right
    grad_X = (1 / m) * (A.T @ (A @ X + B @ U + c @ np.ones(m).T) +
                                 (Lambda - Lambda @ D - D.T @ Lambda) @ X +
                                 D.T @ Lambda @ np.maximum(0, (D @ X + E @ U + f @ np.ones(m).T)) -
                                 Lambda @ (E @ U + f @ np.ones(m).T))
    lip_X = (1 / m) * (np.linalg.norm(A.T @ A + Lambda - Lambda @ D + D.T @ Lambda, ord="fro") +
                       np.max(Lambda) * (np.linalg.norm(D, ord="fro") ** 2))
    return grad_X, lip_X

def Omega_ABc(theta, X, U, Y):
    A, B, c = theta["A"], theta["B"], theta["c"]
    m = np.shape(c)
    assert np.shape(np.outer(c, np.ones(m))) == (m, m)
    return (1 / m) * (A @ X + B @ U + np.outer(c, np.ones(m)) - Y)


def Omega_DEfLambda(m, D, E, f, Lambda, X, U):
    return (1/m) * Lambda @ (np.maximum(0, (D @ X + E @ U + f @ np.ones(m).T)) - X)
