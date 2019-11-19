import numpy as np

def update_theta(theta, X, U, Y):
    grad_theta, lip_ABc, lip_DEf = gradient_descent_theta(theta, X, U, Y)


    for key in grad_theta.keys():
        if key in ["A", "B", "c"]:
            theta[key] -= (1 / lip_ABc) * grad_theta[key]
        elif key in ["D", "E", "f"]:
            theta[key] -= (1/lip_DEf) * grad_theta[key]

    return theta

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

    m = theta["m"]
    AB_Omega = Omega_ABc(theta, X=X, U=U, Y=Y)
    grad_theta["A"] = AB_Omega @ X.T
    grad_theta["B"] = AB_Omega @ U.T
    grad_theta["c"] = AB_Omega @ np.ones(m).reshape(m, 1)

    xnorm = np.linalg.norm(X, ord=2)
    unorm = np.linalg.norm(U, ord=2)
    lip_theta_one = (1 / m) * max(m, xnorm ** 2, unorm ** 2, np.linalg.norm(X @ U.T, ord=2))

    DEF_Omega = Omega_DEfLambda(m, D=theta["D"], E=theta["E"], f=theta["f"],
                                Lambda=theta["Lambda"], X=X, U=U)

    grad_theta["D" ] = DEF_Omega @ X.T
    grad_theta["E"] = DEF_Omega @ U.T
    grad_theta["f"] = DEF_Omega @ np.ones(m).reshape(m, 1)

    lip_theta_two = (np.max(theta["Lambda"]) / m) * max(m, xnorm ** 2, unorm ** 2, xnorm, unorm)

    return grad_theta, lip_theta_one, lip_theta_two

def alpha_x(theta):
    m = theta["m"]
    A = theta["A"]
    D = theta["D"]
    Lambda = theta["Lambda"]

    lip_X = (1 / m) * (np.linalg.norm(A.T @ A + Lambda - Lambda @ D + D.T @ Lambda, ord="fro") +
                       np.max(Lambda) * (np.linalg.norm(D, ord="fro") ** 2))

    return 1 / lip_X

def gradient_descent_x(theta, X, U):
    m = theta["m"]
    A = theta["A"]
    B = theta["B"]
    D = theta["D"]
    E = theta["E"]
    f = theta["f"]
    c = theta["c"]
    Lambda = theta["Lambda"]
    # Make sure that the shapes from the outer products c @ np.ones(m) etc. are right
    grad_X = (1 / m) * (A.T @ (A @ X + B @ U + c @ np.ones(m).reshape(1, m)) +
                                 (Lambda - Lambda @ D - D.T @ Lambda) @ X +
                                 D.T @ Lambda @ np.maximum(0, (D @ X + E @ U + f @ np.ones(m).reshape(1, m))) -
                                 Lambda @ (E @ U + f @ np.ones(m).reshape(1, m)))
    return grad_X

def Omega_ABc(theta, X, U, Y):
    A, B, c = theta["A"], theta["B"], theta["c"]
    m = theta["m"]
    return (1 / m) * (A @ X + B @ U + c @ np.ones(m).reshape(1, m) - Y)


def Omega_DEfLambda(m, D, E, f, Lambda, X, U):
    return (1/m) * Lambda @ (np.maximum(0, (D @ X + E @ U + f @ np.ones(m).reshape(1, m))) - X)
