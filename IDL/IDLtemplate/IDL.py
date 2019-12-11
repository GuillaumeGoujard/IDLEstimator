import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utilities import PicardIterations as pi
import training as idltraining
from utilities import Metrics as me

class IDLModel(BaseEstimator):
    """Implementation of the Scikit-Learn API for Implicit Deep Learning

    Parameters
    ----------
    :param hidden_variables: int
        number of hidden variables of the vector X (see
    :param dual_learning_rate: float
        Positive float, Dual learning rate (IDL's "alpha")
    :param tol_fenchel: float
        Positive float, Fenchel tolerance threshold for dual's update (IDL's "alpha")
    :param inner_tol: float
        Positive float, tolerance threshold for early stopping in the gradient descents with respect to theta
    :param verbosity: bool
        Verbosity of the training process.
    :param random_state: int
        Random number seed for initialization.
    :param initialization_theta

    Note
    ----
    Full documentation of parameters can
        be found here: https://github.com/GuillaumeGoujard/IDLEstimator/blob/master/docs/source/sections/introduction.rst.
    """

    def __init__(self, hidden_variables=1, dual_learning_rate=0.1, tol_fenchel=0.1, inner_tol=1e-3, starting_lambda = None,
                 initialization_theta = None, random_state=0, early_stopping=True,
                 verbosity=True):
        self.is_fitted_ = False
        self.h = hidden_variables
        self.dual_learning_rate, self.tol_fenchtel = dual_learning_rate, tol_fenchel
        self.random_state = random_state
        self.theta, self.training_X = {}, None
        self.verbosity = verbosity
        self.inner_tol = inner_tol
        self.evals_result = []
        self.starting_lambda = starting_lambda
        self.initialization_theta = initialization_theta
        self.early_stopping = early_stopping

    def fit(self, X, y, rounds_number=100, verbose=True, type_of_training="two_loops", eval_set = None):
        """Fit IDL Model \n

        :param X: array_like
            Feature matrix
        :param y: array_like
            Labels
        :param rounds_number: int
            Maximum rounds number in the outer loop
        :param verbose: bool
        :param type_of_training: string
            Two types of training :
                * "two_loops" : RECOMMENDED, we optimize theta and X variables and then we do one step of dual ascent.
                * "one_loop" : one iteration is going to successively operate one step of gradient descent and one step
                of dual ascent

        :return: self : object
            Returns self.
        """
        #For multi-label y, set multi_output=True to allow 2D and sparse y.
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        U = X.T.copy()  # set the shape (n_features, m_samples), to be consistent with IDL notes

        if self.initialization_theta is not None:
            theta, X = self.initialization_theta
        else:
            theta, X = idltraining.initialize_theta_2(U, y, self.h, starting_lambda= self.starting_lambda,
                                                    tol_fenchtel=self.tol_fenchtel, verbose=verbose,
                                                    random_state=self.random_state)

        evals_result = []
        if type_of_training == "two_loops":
            IDLResults = idltraining.train(U, y, theta, X, outer_max_rounds_number=rounds_number,
                                           inner_max_rounds_number=1000,
                                           inner_loop_tol=self.inner_tol, dual_learning_rate=self.dual_learning_rate,
                                           tol_fenchtel=self.tol_fenchtel, evals_result=evals_result, verbose=verbose,
                                           early_stopping = self.early_stopping,
                                           eval_set = eval_set)

        elif type_of_training == "one_loop":
            IDLResults = idltraining.train_theta_lambda(U, y, theta, X, outer_max_rounds_number=rounds_number,
                                                        early_stopping_rounds=20,
                                                        dual_learning_rate=self.dual_learning_rate,
                                                        tol_fenchtel=self.tol_fenchtel, evals_result=evals_result,
                                                        verbose=verbose)
        else:
            print("ERROR")
            return False

        self.theta, self.training_X = IDLResults.theta, IDLResults.X
        self.evals_result = IDLResults.evals_result
        if verbose and self.evals_result is not None:
            me.plot_training_errors(self.evals_result)
        return self,


    def predict(self, X, k_iterations=10000):
        """ Predicting function.
        :param X: array-like
            The input sample.
        :param k_iterations: int
            Maximum number of Picard iterations

        :return: y: array-like
            Returns a prediction array.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        U = X.T.copy()  # set the shape (n_features, m_samples)
        n_features, m_samples = U.shape
        X = np.random.normal(0, 1, (self.h, m_samples))
        X = pi.picard_iterations(X, self.theta["D"], self.theta["E"]@U + self.theta["f"]@np.ones((1, m_samples)),
                                 k_iterations=k_iterations)
        return self.theta["A"]@X + self.theta["B"]@U + self.theta["c"]@np.ones((1, m_samples))

