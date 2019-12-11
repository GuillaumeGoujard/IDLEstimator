from utilities import GradientDescents as gd
from utilities import ConvexProjection as cp
from utilities import DualAscents as da
from utilities import Metrics as me
from utilities import PicardIterations as pi
import numpy as np
import matplotlib.pyplot as plt


class ResultsIDL():
    def __init__(self, theta, X, evals_result=None, **kwargs):
        self.theta = theta
        self.X = X
        self.evals_result = evals_result
        self.__dict__.update(kwargs)


def train(U, y, theta, X, outer_max_rounds_number=50, inner_max_rounds_number=500, inner_loop_tol=1e-3,
          dual_learning_rate=0.1, tol_fenchtel=0.01, evals_result=None, verbose=True,
          early_stopping=True):
    """Train a IDLModel with given parameters.
        Parameters
        ----------
        U : array_like
            Feature matrix.
        y : array_like
            Labels.
        outer_max_rounds_number: int
            Number of iterations in the outer loop, i.e, max number of dual update. Training will stop automatically, if
            the dual variables are not updated for an iteration.
        inner_max_rounds_number: int
            Number of iterations in the inner loop, i.e, max number of steps in gradient descents for theta given dual
            variables.
        inner_loop_tol : float
            Inner loop will break automatically, if the loss does not decrease by more than the inner_loop_tol for an
            iteration
        dual_learning_rate : float
            Positive float, Dual learning rate (IDL's "alpha")
        tol_fenchtel : float
            Positive float, Fenchtel tolerance threshold for dual's update (IDL's "alpha")
        evals_result: list of pairs (float, float)
            Training losses list for general loss and L2 Loss. Will help us track the performance of the model.
        verbose : bool
            If **verbose** is True then the evaluation metric on the training set is
            printed at each dual update stage.
        solver : string
            solver used for cvxpy, if None then we select "ECOS"

        Returns
        -------
        ResultsIDL : a trained IDL model
    """

    L = me.loss(U, y, theta, X)
    if evals_result is not None:
        evals_result.append([L, me.L2Loss(U, y, theta, X), np.linalg.norm(theta["Lambda"], ord=1),
                             me.how_far_from_RELU(U, X, theta)])

    if verbose:
        print("Launching training... \n")
    i = 0
    for j in range(outer_max_rounds_number):
        for k in range(inner_max_rounds_number):
            theta = gd.update_theta(theta, X, U, y)
            theta["D"] = cp.non_cvx_projection(theta["D"], epsilon=1e-1)
            X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U, y))
            nL = me.loss(U, y, theta, X)
            l = me.L2Loss(U, y, theta, X)
            if evals_result is not None:
                evals_result.append([nL, l, np.linalg.norm(theta["Lambda"], ord=1), me.how_far_from_RELU(U, X, theta)])
            if abs(nL - L) < inner_loop_tol:
                break
            L = nL

        last_lambda = np.diag(theta["Lambda"])
        updated_lambda_vector = da.update_dual(U, theta, X, alpha=dual_learning_rate, epsilon=tol_fenchtel)
        dlambda = updated_lambda_vector - last_lambda
        theta["Lambda"] = np.diag(updated_lambda_vector)

        nL = me.loss(U, y, theta, X)
        l = me.L2Loss(U, y, theta, X)
        if evals_result is not None:
            evals_result.append([nL, l, np.linalg.norm(theta["Lambda"], ord=1), me.how_far_from_RELU(U, X, theta)])

        if verbose :
            print("Updating Lambda : General Loss after round {} : ".format(j + 1), round(nL, 3), " L2Loss : ",
                  round(l, 3))
            if j % 10 == 0 :
                pi.plot_before_after_picard(U, y, theta, X, j)

        if (dlambda == np.zeros(dlambda.shape)).all() and early_stopping:
            print("finished training !")
            break

    training_result = ResultsIDL(theta, X, evals_result=evals_result)
    return training_result



def train_theta_lambda(U, y, theta, X, outer_max_rounds_number=50, early_stopping_rounds=10, dual_learning_rate=0.1,
                       tol_fenchtel=0.01, evals_result=None, verbose=True, solver=None, solver_options=None):
    """Train a IDLModel with given parameters.
        Parameters
        ----------
        U : array_like
            Feature matrix.
        y : array_like
            Labels.
        outer_max_rounds_number: int
            Number of iterations in the outer loop.
        early_stopping_rounds: int
            Training will stop automatically if for early_stopping_rounds iterations, the general loss has not decreased.
        dual_learning_rate : float
            Positive float, Dual learning rate (IDL's "alpha")
        tol_fenchtel : float
            Positive float, Fenchtel tolerance threshold for dual's update (IDL's "alpha")
        evals_result: list of pairs (float, float)
            Training losses list for general loss and L2 Loss. Will help us track the performance of the model.
        verbose : bool
            If **verbose** is True then the evaluation metric on the training set is
            printed at each dual update stage.
        solver : string
            solver used for cvxpy, if None then we select "ECOS"

        Returns
        -------
        ResultsIDL : a trained IDL model
    """

    L = me.loss(U, y, theta, X)
    if solver_options is None:
        solver_options = {}
    best_theta_yet = (L, theta, X)
    if verbose:
        print("Launching training... \n")
    i = 0
    for k in range(outer_max_rounds_number):
        theta = gd.update_theta(theta, X, U, y)
        theta["D"] = cp.non_cvx_projection(theta["D"], epsilon=1e-1)
        X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U, y))
        theta["Lambda"] = np.diag(da.update_dual(U, theta, X, alpha=dual_learning_rate, epsilon=tol_fenchtel))
        nL = me.loss(U, y, theta, X)
        if early_stopping_rounds is not None:
            if nL > L:
                i = i + 1
                if i > early_stopping_rounds:
                    if verbose:
                        print("The general loss is increasing : early stopping...")
                    break
            else:
                i = 0
        l = me.L2Loss(U, y, theta, X)
        if evals_result is not None:
            evals_result.append([nL, l])
        if verbose:
            print("General Loss for round {} : ".format(k), round(nL, 3), " L2Loss : ", round(l, 3))
        L = nL
        if k == 1:
            best_theta_yet = (nL, theta, X)
        if nL < best_theta_yet[0]:
            best_theta_yet = (nL, theta, X)

        if verbose :
            print("Updating Lambda : General Loss after round {} : ".format(k+1), round(nL, 3), " L2Loss : ", round(l, 3))

    theta = best_theta_yet[1]
    X = best_theta_yet[2]

    training_result = ResultsIDL(theta, X, evals_result=evals_result)
    return training_result

    
    
def initialize_theta_2(U, y, h_variables, tol_fenchtel=0.01, verbose=True, random_state=0, starting_lambda=None):
    """Initialize theta and X for a IDLModel with given parameters.
            Parameters
            ----------
            U : array_like
                Feature matrix.
            y : array_like
                Labels.
            h_variables: int
                Number of hidden variables in the X vector.
            dual_learning_rate : float
                Positive float, Dual learning rate (IDL's "alpha").
            tol_fenchtel : float
                Positive float, Fenchtel tolerance threshold for dual's update (IDL's "alpha").
            verbose : bool
                If **verbose** is True then some metrics will be printed on the console.
            random_state : int
                Random number seed.

            Returns
            -------
            Theta, X : a pair of IDL parameter and hidden features
    """
    
    if verbose:  
        print("=" * 60)
        print("Initialization")
        
    n, m = U.shape
    if len(y.shape) == 1:  # We need this to solve the problem (m_samples, ) or (m_samples, p_outputs)
        p = 1
    else:
        _, p = y.shape

    h = h_variables
    np.random.RandomState(random_state)

    A = np.random.normal(0, 1, (p, h))
    B = np.random.normal(0, 1, (p, n))
    c = np.random.normal(0, 1, (p, 1))
    D = np.random.normal(0, 1, (h, h))
    E = np.random.normal(0, 1, (h, n))
    f = np.random.normal(0, 1, (h, 1))
    X = np.random.normal(0, 1, (h, m))
    X[X < 0] = 0
    lambda_dual = np.ones((h_variables))
    Lambda = np.diag(lambda_dual)
    theta = {"A": A, "B": B, "c": c, "D": D, "E": E, "f": f, "Lambda": Lambda, "m": m}

    theta["D"] = cp.non_cvx_projection(theta["D"], epsilon=0.5)
    X = pi.picard_iterations(X, theta["D"], theta["E"] @ U + theta["f"] @ np.ones((1, theta["m"])),
                             k_iterations=1000)
    for k in range(1):
        theta = gd.block_update_ABc(U,y,theta, X)
        # X = block_update_X(U,y,theta, X, num_iter)
        # theta = block_update_DEf(U, y, theta, X, 3*num_iter)

    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    y_fenchtel = D @ X + E @ U + f @ np.ones((1, m))
    fenchel = me.fenchtel_div(X, y_fenchtel)
    if starting_lambda is None:
        first_lambda = (me.loss(U,y, theta, X))/np.linalg.norm(fenchel)
    else:
        first_lambda = starting_lambda
    print("FIRST LAMBDA = ", first_lambda)
    theta["Lambda"] = np.diag(np.ones(h))*first_lambda
    theta["Lambda"] = np.diag(da.update_dual(U, theta, X, alpha=1, epsilon=tol_fenchtel))
    print("Initialization is a Success ! ")
    print("...")

    plt.scatter(U[0,:],y)
    plt.scatter(U[0,:],theta["A"]@X + theta["B"]@U + theta["c"]@np.ones((1, theta["m"])))
    print(np.linalg.norm(X-np.maximum(0, theta["D"] @ X + theta["E"] @ U + theta["f"]@np.ones((1, theta["m"]))), ord= "fro"))
    Xpic = pi.picard_iterations(X, theta["D"], theta["E"]@U + theta["f"]@np.ones((1,theta["m"])), k_iterations=1000)
    plt.scatter(U[0,:],theta["A"]@Xpic + theta["B"]@U + theta["c"]@np.ones((1, theta["m"])), label = "pic")
    plt.legend()
    plt.title("initialization h = " + str(h))
    plt.show()
    
    return theta, X


