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
    tmp = cp.bmat([[np.diag(np.repeat(1-epsilon, h)), proj_D], [proj_D.T, np.diag(np.repeat(1-epsilon, h))]])
    constraint = [expression >= 0] + [tmp >= 0]
    objective = cp.norm(D - proj_D, p="fro")
    problem = cp.Problem(cp.Minimize(objective), constraints=constraint)
    problem.solve(verbose=False, solver=solver, warm_start=True, **kwargs)
    theta["D"] = proj_D.value

    return theta



def S_theta(theta):
    #TODO
    pass
