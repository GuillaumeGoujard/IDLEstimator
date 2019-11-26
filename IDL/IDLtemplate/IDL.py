import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utilities import GradientDescents as gd
from utilities import ConvexProjection as cp
from utilities import DualAscents as da
from utilities import Metrics as me
from utilities import PicardIterations as pi
import matplotlib.pyplot as plt


class IDLModel(BaseEstimator):
    """
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, hidden_features=1, alpha=0.1, epsilon=0.1, random_state=0, seed=None):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        self.is_fitted_ = False
        self.theta = {}
        self.h = hidden_features
        self.training_X = None
        self.alpha, self.epsilon = alpha, epsilon




    def fit(self, X, y, verbose=1, rounds_number=100,
            sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, callbacks=None ):
        """
        Fit the IDLModel parameters

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (m_samples,  n_features)
            The training input samples.
        y : array-like, shape (m_samples, ) or (m_samples, p_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        verbose: If True (1) then we print everything on the console
        rounds_number : how many rounds max do you want to do
        early_stopping : If True : once the L2Loss will increase we automatically stop
        sample_weight : array_like
                    instance weights
        eval_set : list, optional
                    A list of (X, y) tuple pairs to use as a validation set for
                    early-stopping
        eval_metric : str, callable, optional
                    If a str, should be a built-in evaluation metric to use. See
                    doc/parameter.rst. If callable, a custom evaluation metric. The call
                    signature is func(y_predicted, y_true) where y_true will be a
                    DMatrix object such that you may need to call the get_label
                    method. It must return a str, value pair where the str is a name
                    for the evaluation and value is the value of the evaluation
                    function. This objective is always minimized.
        early_stopping_rounds : int
                    Activates early stopping. Validation error needs to decrease at
                    least every <early_stopping_rounds> round(s) to continue training.
                    Requires at least one item in evals.  If there's more than one,
                    will use the last. Returns the model from the last iteration
                    (not the best one). If early stopping occurs, the model will
                    have three additional fields: bst.best_score, bst.best_iteration
                    and bst.best_ntree_limit.
                    (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
                    and/or num_class appears in the parameters)

        Returns
        -------
        self : object
            Returns self.
        """
        #For multi-label y, set multi_output=True to allow 2D and sparse y.
        #Standard sklearn fit is : X : (m_samples,  n_features) and y : (m_samples, p_outputs)
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.is_fitted_ = True
        U = X.T.copy() #set the shape (n_features, m_samples)
        n_features, m_samples = U.shape
        if len(y.shape) == 1: #We need this to solve the problem (m_samples, ) or (m_samples, p_outputs)
            p_outputs = 1
        else:
            _, p_outputs = y.shape
        self.theta, X = initialize_theta(y, U, n_features, m_samples, p_outputs, self.h, verbose=verbose, alpha=self.alpha,
                                         epsilon=self.epsilon)
        theta = self.theta
        training_errors = [] #we are going to store the drop in the loss of our training
        L = me.loss(y, U, theta, X)
        training_errors.append([L, me.L2Loss(y, U, theta, X)])

        best_theta_yet = (L, theta)
        if verbose:
            print("Launching training... \n")
        i = 0 #i = number of time that the loss has not decreased for early stopping
        for k in range(rounds_number):
            nL = me.loss(y, U, theta, X)  # We will have to discuss about this, normally we should aim at decreasing the general loss
            if early_stopping_rounds is not None:
                # but it seemed that the early stopping functionned better with the L2Loss
                if nL > L:
                    i = i+1
                    if i > early_stopping_rounds:
                        if verbose:
                            print("The general loss is increasing : early stopping...")
                        break
                else:
                    i = 0
            l = me.L2Loss(y, U, theta, X)
            training_errors.append([nL, l])
            if verbose:
                print("General Loss for round {} : ".format(k), round(nL, 3), " L2Loss : ", round(l, 3))
            theta = gd.update_theta(theta, X, U, y)
            theta = cp.project_to_S_theta(theta)
            X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U))
            theta["Lambda"], dlambda = np.diag(da.update_dual(theta, X, U, alpha=self.alpha, epsilon=self.epsilon))
            L = nL
            if k == 1:
                best_theta_yet = (nL, theta)
            if nL < best_theta_yet[0]:
                best_theta_yet = (nL, theta)

        self.theta = best_theta_yet[1]
        self.training_X = X #We keep the X in memory
        if verbose:
            print("="*70)
            print("Returned theta for general loss : ", best_theta_yet[0])
            # print("RESULTS")
            # print("A@X + B@U + c = ", self.theta["A"]@self.training_X + self.theta["B"]@U + self.theta["c"]@np.ones((1,self.theta["m"])))
            # print("y = ", y)
            # print("(D@X + E@U + f@1m)+ = ", np.maximum(0, self.theta["D"]@self.training_X+self.theta["E"]@U+
            #                                            self.theta["f"]@np.ones((1,self.theta["m"]))))
            # print("X = ", self.training_X)
            print("=" * 70)
            plot_training_errors(training_errors)

        # `fit` should always return `IDL`
        return self,


    def fit2(self, X, y, verbose=1, rounds_number=100,
            sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, callbacks=None ):
        """
        Fit the IDLModel parameters

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (m_samples,  n_features)
            The training input samples.
        y : array-like, shape (m_samples, ) or (m_samples, p_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        verbose: If True (1) then we print everything on the console
        rounds_number : how many rounds max do you want to do
        early_stopping : If True : once the L2Loss will increase we automatically stop
        sample_weight : array_like
                    instance weights
        eval_set : list, optional
                    A list of (X, y) tuple pairs to use as a validation set for
                    early-stopping
        eval_metric : str, callable, optional
                    If a str, should be a built-in evaluation metric to use. See
                    doc/parameter.rst. If callable, a custom evaluation metric. The call
                    signature is func(y_predicted, y_true) where y_true will be a
                    DMatrix object such that you may need to call the get_label
                    method. It must return a str, value pair where the str is a name
                    for the evaluation and value is the value of the evaluation
                    function. This objective is always minimized.
        early_stopping_rounds : int
                    Activates early stopping. Validation error needs to decrease at
                    least every <early_stopping_rounds> round(s) to continue training.
                    Requires at least one item in evals.  If there's more than one,
                    will use the last. Returns the model from the last iteration
                    (not the best one). If early stopping occurs, the model will
                    have three additional fields: bst.best_score, bst.best_iteration
                    and bst.best_ntree_limit.
                    (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
                    and/or num_class appears in the parameters)

        Returns
        -------
        self : object
            Returns self.
        """
        #For multi-label y, set multi_output=True to allow 2D and sparse y.
        #Standard sklearn fit is : X : (m_samples,  n_features) and y : (m_samples, p_outputs)
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.is_fitted_ = True
        U = X.T.copy() #set the shape (n_features, m_samples)
        n_features, m_samples = U.shape
        if len(y.shape) == 1: #We need this to solve the problem (m_samples, ) or (m_samples, p_outputs)
            p_outputs = 1
        else:
            _, p_outputs = y.shape
        self.theta, X = initialize_theta(y, U, n_features, m_samples, p_outputs, self.h, verbose=verbose, alpha=self.alpha,
                                         epsilon=self.epsilon)
        theta = self.theta
        training_errors = [] #we are going to store the drop in the loss of our training
        L = me.loss(y, U, theta, X)
        training_errors.append([L, me.L2Loss(y, U, theta, X)])

        best_theta_yet = (L, theta)
        if verbose:
            print("Launching training... \n")
        # i = 0 #i = number of time that the loss has not decreased for early stopping
        for j in range(int(rounds_number/10)):
            for k in range(rounds_number*10):
                nL = me.loss(y, U, theta, X)
                l = me.L2Loss(y, U, theta, X)
                training_errors.append([nL, l])
                theta = gd.update_theta(theta, X, U, y)
                theta = cp.project_to_S_theta(theta)
                X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X, U))
                if abs(nL - me.loss(y, U, theta, X)) < self.epsilon:
                    break
            nL = me.loss(y, U, theta, X)
            l = me.L2Loss(y, U, theta, X)
            if verbose:
                print("Updating Lambda : General Loss for round {} : ".format(j), round(nL, 3), " L2Loss : ", round(l, 3))
                print("Fenchel:", me.fenchtel_div(X, theta["D"] @ X + theta["E"] @ U + theta["f"].reshape(
                    (theta["f"].shape[0], 1)) @ np.ones((1, theta["m"]))))
                print("Lambda : ", np.diag(theta["Lambda"]))
            lambda_vector, dlambda = da.update_dual(theta, X, U, alpha=self.alpha, epsilon=self.epsilon)
            theta["Lambda"] = np.diag(lambda_vector)
            if (dlambda == np.zeros(dlambda.shape)).all():
                print("finished training !")
                break

        # self.theta = best_theta_yet[1]
        self.training_X = X #We keep the X in memory
        if verbose:
            print("="*70)
            print("Returned theta for general loss : ", nL)
            # print("RESULTS")
            # print("A@X + B@U + c = ", self.theta["A"]@self.training_X + self.theta["B"]@U + self.theta["c"]@np.ones((1,self.theta["m"])))
            # print("y = ", y)
            # print("(D@X + E@U + f@1m)+ = ", np.maximum(0, self.theta["D"]@self.training_X+self.theta["E"]@U+
            #                                            self.theta["f"]@np.ones((1,self.theta["m"]))))
            # print("X = ", self.training_X)
            print("=" * 70)
            plot_training_errors(training_errors)

        # `fit` should always return `IDL`
        return self,

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        U = X.T.copy()  # set the shape (n_features, m_samples)
        n_features, m_samples = U.shape
        X = np.random.normal(0, 1, (self.h, m_samples))
        X = pi.picard_iterations(X, self.theta["D"], self.theta["E"]@U + self.theta["f"]@np.ones((1, m_samples)),
                             k_iterations=100)
        return self.theta["A"]@X + self.theta["B"]@U + self.theta["c"]@np.ones((1, m_samples))



class IDLClassifier(IDLModel):
    def __init__(self, hidden_features=1, alpha=0.1, epsilon=0.1, random_state=0, seed=None):
        super(IDLModel, self).__init__(hidden_features=hidden_features, alpha=alpha,
                                       epsilon=epsilon, random_state=random_state, seed=seed)




def initialize_theta(y, U, n_features, m_samples, p_outputs, h_variables, alpha=0.1, epsilon=0.1, verbose=True):
    """

    :param y:
    :param U:
    :param n_features:
    :param m_samples:
    :param p_outputs:
    :param h_variables:
    :param alpha:
    :param epsilon:
    :param verbose:
    :return:
    """

    """
    Create random instance of theta
    """
    print("=" * 60)
    print("Initialization")

    n, m = n_features, m_samples
    p = p_outputs
    h = h_variables
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


    """
    Best X, A, B, c guesses
    """
    print("_" * 50)
    print("Best X, A, B, c guesses")
    L = me.L2Loss(y, U, theta, X)
    best_theta_yet = (L, theta)
    rounds_number, early_stopping_rounds = 300, 20
    i = 0
    for k in range(rounds_number):
        nL = me.L2Loss(y, U, theta, X)
        if early_stopping_rounds is not None:
            if nL > L:
                i = i + 1
                if i > early_stopping_rounds:
                    if verbose:
                        print("The L2 loss is increasing : early stopping...")
                    break
            else:
                i = 0
        # if verbose:
        #     print("Initialization A,B,c,X : L2 Loss for round {} : ".format(k), round(nL, 3))
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
            # but it seemed that the early stopping functionned better with the L2Loss
            if nL > L:
                i = i + 1
                if i > early_stopping_rounds:
                    if verbose:
                        print("The Fenchtel loss is increasing : early stopping...")
                    break
            else:
                i = 0
        # if verbose:
        #     print("Initialization D,E,f : Fenchtel Loss for round {} : ".format(k), round(nL, 3))
        theta = gd.update_DEf_init(theta, X, U, y)
        if nL < best_theta_yet[0]:
            best_theta_yet = (nL, theta)
    theta = best_theta_yet[1]

    lambda_vector, dlambda = da.update_dual(theta, X, U, alpha=alpha, epsilon=epsilon)
    theta["Lambda"] = np.diag(epsilon*np.ones(h))
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
