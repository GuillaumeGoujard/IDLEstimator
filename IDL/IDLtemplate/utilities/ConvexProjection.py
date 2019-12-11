import cvxpy as cp, numpy as np

import cvxpy as cp, numpy as np

def project_to_S_theta(theta, epsilon=.05, solver=None, **kwargs):
    """"
    This function projects theta to the convex set
    Lambda - (Lambda @ D + D^T @ Lambda) PSD

    """
    Lambda = theta["Lambda"]
    proj_D = cp.Variable(shape=theta["D"].shape)
    h, h = theta["D"].shape

    D = theta["D"]

    expression = Lambda - (Lambda @ proj_D + proj_D.T @ Lambda)
    tmp = cp.bmat([[np.diag(np.repeat(1 - epsilon, h)), proj_D], [proj_D.T, np.diag(np.repeat(1 - epsilon, h))]])
    constraint = [expression >= 0] + [tmp >= 0]
    objective = cp.norm(D - proj_D, p="fro")
    problem = cp.Problem(cp.Minimize(objective), constraints=constraint)
    problem.solve(verbose=False, solver=solver, warm_start=True, **kwargs)
    theta["D"] = proj_D.value

    return theta


def non_cvx_projection(D, epsilon=1e-3):
    h,h = D.shape
    for i in range(h):
        D[i, :] = our_cvx_proj(D[i, :] , ballRadius=1-epsilon)
    return D


def our_cvx_proj(vY, ballRadius=1.0, stopThr=1e-6):
    if np.sum(np.abs(vY)) <= ballRadius:
        return vY
    paramLambda = 0
    objVal = np.sum(np.max(np.abs(vY) - paramLambda, 0)) - ballRadius

    while (np.abs(objVal) > stopThr):
        objVal = np.sum(np.maximum(np.abs(vY) - paramLambda, 0)) - ballRadius
        df = np.sum(-1*((np.abs(vY) - paramLambda) > 0))
        paramLambda = paramLambda - (objVal / df)

    paramLambda = np.max(paramLambda, 0)

    vX = np.sign(vY) * np.maximum(np.abs(vY) - paramLambda, 0)
    return vX

def S_theta(theta):
    # TODO
    pass


"""
CREDTI: https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
"""


