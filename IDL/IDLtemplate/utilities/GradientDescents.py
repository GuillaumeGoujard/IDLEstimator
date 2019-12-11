import numpy as np
from utilities import ConvexProjection as cp

def update_theta(theta, X, U, Y):
    grad_theta, lip_ABc, lip_DEf = gradient_descent_theta(theta, X, U, Y)

    for key in grad_theta.keys():
        if key in ["A", "B", "c"]:
            theta[key] -= (1 / lip_ABc) * grad_theta[key]
        elif key in ["D", "E", "f"]:
            theta[key] -= (1/lip_DEf) * grad_theta[key]

    return theta


def update_ABc_init(theta, X, U, Y):
    grad_theta, lip_ABc, lip_DEf = gradient_descent_theta(theta, X, U, Y)
    for key in grad_theta.keys():
        if key in ["A", "B", "c"]:
            theta[key] -= (1 / lip_ABc) * grad_theta[key]
    return theta


def update_DEf_init(theta, X, U, Y):
    theta["Lambda"] = np.identity(theta["Lambda"].shape[0])
    grad_theta, lip_ABc, lip_DEf = gradient_descent_theta(theta, X, U, Y)
    for key in grad_theta.keys():
        if key in ["D", "E", "f"]:
            theta[key] -= (1 / lip_DEf) * grad_theta[key]
    return theta


def gradient_descent_theta(theta, X, U, Y):
    """
    Returns the gradient of theta \n
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


def alpha_x(theta, initialization=False):
    m = theta["m"]
    A = theta["A"]
    D = theta["D"]
    Lambda = theta["Lambda"]

    if initialization:
        lip_X = (1 / m) * (np.linalg.norm(A.T @ A))
    else:
        lip_X = (1 / m) * (np.linalg.norm(A.T @ A + Lambda - Lambda @ D + D.T @ Lambda, ord="fro") +
                           np.max(Lambda) * (np.linalg.norm(D, ord="fro") ** 2))
    return 1 / lip_X

def gradient_descent_x(theta, X, U, y, initialization=False):
    m = theta["m"]
    A = theta["A"]
    B = theta["B"]
    D = theta["D"]
    E = theta["E"]
    f = theta["f"]
    c = theta["c"]
    Lambda = theta["Lambda"]
    # Make sure that the shapes from the outer products c @ np.ones(m) etc. are right
    if initialization :
        grad_X = (1 / m) * (A.T @ (A @ X + B @ U + c @ np.ones(m).reshape(1, m) - y ))
    else:
        grad_X = (1 / m) * (A.T @ (A @ X + B @ U + c @ np.ones(m).reshape(1, m) - y) +
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


def block_update_X(U, y, theta, X, num_iter):
    for k in range(num_iter):
        X = np.maximum(0, X - alpha_x(theta) * gradient_descent_x(theta, X, U, y, initialization=True))
    return X


def block_update_ABc(U, y, theta, X):
    h = theta["A"].shape[1]
    n = theta["B"].shape[1]

    Z = np.concatenate((X, U, np.ones((1, theta["m"]))), axis=0)
    Theta = np.linalg.solve(Z @ Z.T + np.diag(np.ones((h + n + 1)) * 0.000001), Z @ y)
    if (len(Theta.shape) == 1):
        theta["A"] = Theta[0:h].reshape((1, h))
        theta["B"] = Theta[h:h + n].reshape((1, n))
        theta["c"] = Theta[h + n].T.reshape((1, 1))
    else:
        theta["A"] = Theta[0:h, :].T
        theta["B"] = Theta[h:h + n:, :].T
        theta["c"] = Theta[h + n, :].T

    return theta


def block_update_DEf(U, y, theta, X, num_iter):
    for k in range(num_iter):
        theta = update_DEf_init(theta, X, U, y)
        theta["D"] = cp.non_cvx_projection(theta["D"], epsilon=1e-1)
    return theta

