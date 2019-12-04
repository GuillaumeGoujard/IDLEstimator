from utilities import GradientDescents as gd
from utilities import ConvexProjection as cp
from utilities import DualAscents as da
from utilities import Metrics as me
from scipy import sparse
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
          dual_learning_rate=0.1, tol_fenchtel=0.01, evals_result=None, verbose=True, solver=None, solver_options=None):
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
    if solver_options is None:
        solver_options = {}

    if evals_result is not None:
        evals_result.append([L, me.L2Loss(U, y, theta, X)])

    if verbose:
        print("Launching training... \n")
    i = 0  # i = number of time that the loss has not decreased for early stopping
    for j in range(outer_max_rounds_number):
        for k in range(inner_max_rounds_number):
            theta = gd.update_theta(theta, X, U, y)
            theta["D"] = cp.non_cvx_projection(theta["D"], epsilon=1e-1)
            X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U))

            nL = me.loss(U, y, theta, X)
            l = me.L2Loss(U, y, theta, X)
            if evals_result is not None:
                evals_result.append([nL, l])
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
            evals_result.append([nL, l])
        if verbose:
            rounding = int(np.log((1/inner_loop_tol)))
            print("Updating Lambda : General Loss for round {} : ".format(j), round(nL, rounding), " L2Loss : ",
                  round(l, rounding))
            # print("Fenchel:", me.fenchtel_div(X, theta["D"] @ X + theta["E"] @ U + theta["f"].reshape(
            #     (theta["f"].shape[0], 1)) @ np.ones((1, theta["m"]))))
            # print("Lambda : ", np.diag(theta["Lambda"]))

        if (dlambda == np.zeros(dlambda.shape)).all():
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
        theta = cp.project_to_S_theta(theta, solver=solver, **solver_options)
        X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U))
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

    theta = best_theta_yet[1]
    X = best_theta_yet[2]

    training_result = ResultsIDL(theta, X, evals_result=evals_result)
    return training_result



def initialize_theta(U, y, h_variables, dual_learning_rate=0.1, tol_fenchtel=0.01, verbose=True, random_state=0):
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
    A = sparse.random(p, h).toarray()
    B = sparse.random(p, n).toarray()
    c = sparse.random(p, 1).toarray()
    D = sparse.random(h, h).toarray()
    D = cp.non_cvx_projection(D, epsilon=1e-3)
    E = sparse.random(h, n).toarray()
    f = sparse.random(h, 1).toarray()
    # A = np.random.normal(0, 1, (p, h))
    # B = np.random.normal(0, 1, (p, n))
    # c = np.random.normal(0, 1, (p, 1))
    # D = np.random.normal(0, 1, (h, h))
    # E = np.random.normal(0, 1, (h, n))
    # f = np.random.normal(0, 1, (h, 1))
    X = np.random.normal(0, 1, (h, m))
    X[X < 0] = 0
    lambda_dual = np.ones((h_variables))
    Lambda = np.diag(lambda_dual)
    theta = {"A": A, "B": B, "c": c, "D": D, "E": E, "f": f, "Lambda": Lambda, "m": m}

    """
    Best X, A, B, c guesses
    """
    print("_" * 50)
    print("Best X, A, B, c guesses")
    L = me.L2Loss(U, y, theta, X)
    best_theta_yet = (L, theta)
    rounds_number, early_stopping_rounds = 300, 20
    i = 0
    for k in range(rounds_number):
        nL = me.L2Loss(U, y, theta, X)
        if early_stopping_rounds is not None:
            if nL > L:
                i = i + 1
                if i > early_stopping_rounds:
                    if verbose:
                        print("The L2 loss is increasing : early stopping...")
                    break
            else:
                i = 0
        theta = gd.update_ABc_init(theta, X, U, y)
        X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U, initialization=True))
        if nL < best_theta_yet[0]:
            best_theta_yet = (nL, theta)
    theta = best_theta_yet[1]

    """
    Best D, E, f guesses
    """
    print("_"*50)
    print("Best D, E, f guesses")
    L = me.fenchtel_error(theta, X, U, lambda_=np.ones((1, h)))
    best_theta_yet = (L, theta)
    rounds_number, early_stopping_rounds = 300, 20
    i = 0
    for k in range(rounds_number):
        nL = me.fenchtel_error(theta, X, U, lambda_=np.ones((1, h)))
        if early_stopping_rounds is not None:
            if nL > L:
                i = i + 1
                if i > early_stopping_rounds:
                    if verbose:
                        print("The Fenchtel loss is increasing : early stopping...")
                    break
            else:
                i = 0
        theta = gd.update_DEf_init(theta, X, U, y)
        if nL < best_theta_yet[0]:
            best_theta_yet = (nL, theta)
    theta = best_theta_yet[1]
    lambda_vector = da.update_dual(U, theta, X, alpha=dual_learning_rate, epsilon=tol_fenchtel)
    theta["Lambda"] = np.diag(tol_fenchtel*np.ones(h))
    print("Initialization is a Success ! ")
    print("...")
    return theta, X


def plot_training_errors(training_errors):
    L2_loss = np.array(training_errors)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('rounds')
    ax1.set_ylabel('General Loss', color=color)
    ax1.plot(L2_loss[:, 0][1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('L2 loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(L2_loss[:, 1][1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
