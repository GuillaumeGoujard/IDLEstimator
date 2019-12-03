import cvxpy as cp, numpy as np


def project_to_S_theta(theta, epsilon=.05):
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

    #symm_mtrx = cp.Variable(shape=theta["Lambda"].shape, PSD=True)
    # We now reformulate the norm constraint on D to a LMI through Schur Complement

    problem = cp.Problem(cp.Minimize(objective), constraints=constraint)

    # try:
    problem.solve(verbose=True, solver="SCS")
    # except:
    #     print("PROBLEM")
    #     problem.solve(verbose=True, solver="SCS")

    # try:
    #     problem.solve(verbose=False)
    # except:
    #     problem.solve(verbose=True)
    theta["D"] = proj_D.value

    return theta



def S_theta(theta):
    #TODO
    pass
