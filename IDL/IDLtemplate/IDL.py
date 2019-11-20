import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utilities import GradientDescents as gd
from utilities import ConvexProjection as cp
from utilities import DualAscents as da
from utilities import Metrics as me

class IDLEstimator(BaseEstimator):
    """
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, hidden_features=1, demo_param='demo_param'):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        self.demo_param = demo_param
        self.is_fitted_ = False
        self.theta = []
        self.h = hidden_features



    def fit(self, X, y):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        U = X.copy()

        """
        Initialization of the theta vector
        """
        n, N = U.shape
        p, = y.shape
        A = np.ones((p, self.h))
        B = np.ones((p, n))
        c = np.ones((p))
        D = np.ones((self.h, self.h))
        E = np.ones((self.h, n))
        f = np.ones((self.h))
        X = None
        lambda_dual = np.ones((self.h))
        Lambda = np.diag(lambda_dual)
        theta = {"A": A, "B": B, "c": c, "D": D, "E": E, "f": f, "Lambda": Lambda}

        for k in range(100):
            L = me.Loss(y, X, theta, U)
            theta = gd.update_theta(theta, X, U, Y)
            #theta = theta - gd.alpha_theta(X, U)@gd.gradient_descent_theta(X, theta, U, y)
            theta = cp.project_to_S_theta(theta)
            X = np.maximum(0, X - gd.alpha_x(theta)@gd.gradient_descent_x(X, theta, U, y))
            lambda_dual = da.update_dual(theta, X, U)

        self.theta = theta
        # `fit` should always return `self`
        return self

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
        return np.ones(X.shape[0], dtype=np.int64)

