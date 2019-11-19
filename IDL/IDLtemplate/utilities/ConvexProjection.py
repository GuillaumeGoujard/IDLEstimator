import cvxpy as cp, numpy as np


def project_to_S_theta(theta, epsilon=.05):
    """"
    This function projects to the convex set
    Lambda - (Lambda @ D + D^T @ Lambda)
    instead of
    Lambda + A^TA - (Lambda @ D + D^T @ Lambda),
    since the latter is not convex in the variables (A, D)
    """

    Lambda = theta["Lambda"]
    proj_A = cp.Variable(shape=theta["A"].shape, value=theta["A"])
    proj_D = cp.Variable(shape=theta["D"].shape, value=theta["D"])
    h, h = theta["D"].shape

    #
    #big_D = cp.Variable(value=np.block([[(1 - epsilon) * np.eye(h), theta["D"]],
    #                  [theta["D"].T, (1 - epsilon) * np.eye(h)]]))

    # We only consider big_D as an optimization variable and not only D,
    # Due to difficulties of implementing the PSD constraint on big_D while
    # only keeping D as a cvxpy variable
    #tmp_expression = proj_Lambda + (proj_A.T) @ (proj_A) - (proj_Lambda @ big_D.value[h:, :h] + big_D.value[h:, :h].T @ proj_Lambda)

    #first, second = theta["D"], theta["A"]

    # Since Lambda is diagonal, the matrix multiplication below could be more efficient right?
    # -> could just sum last two terms and multiply by lambda from either the right or left

    expression = Lambda - (Lambda @ proj_D + proj_D.T @ Lambda)  # + (proj_A.T) @ (proj_A)

    symm_mtrx = cp.Variable(shape=theta["Lambda"].shape, PSD=True)

    # We now reformulate the norm constraint on D to a LMI through Schur Complement

    tmp = cp.bmat([[np.diag(np.repeat(1-epsilon, h)), proj_D], [proj_D.T, np.diag(np.repeat(1-epsilon, h))]])

    objective = cp.norm(symm_mtrx - expression, p="fro")

    problem = cp.Problem(cp.Minimize(objective), constraints=[tmp >= 0])

    problem.solve()

    # etteren, etterto = proj_D.value, proj_A.value

    theta["D"] = proj_D.value
    theta["A"] = proj_A.value

    return theta



def S_theta(theta):
    #TODO
    pass
