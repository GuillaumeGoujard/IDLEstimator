import IDL as idl
import numpy as np
from utilities import PicardIterations as pi
from utilities import ConvexProjection as cp
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import training as idltraining

def generate_a_model():
    n, m = 10, 10000
    p = 1
    h = 500
    np.random.RandomState(0)
    A = sparse.random(p, h).toarray()
    B = sparse.random(p, n).toarray()
    c = sparse.random(p, 1).toarray()
    D = sparse.random(h, h).toarray()
    D = cp.non_cvx_projection(D, epsilon=1e-3)
    E = sparse.random(h, n).toarray()
    f = sparse.random(h, 1).toarray()
    U = np.random.normal(0, 1, (n,m))
    X = np.random.normal(0, 1, (h, m))

    X = pi.picard_iterations(X, D, E@U + f@np.ones((1, m)),
                                     k_iterations=1000)
    y = A@X + B@U + c@np.ones((1, m))
    return U, y[0]


U, hat_y = generate_a_model()
X = U.T

test_data_point = int(0.1*X.shape[0])
X_train, y_train, X_test, y_test = X[:-test_data_point], hat_y[:-test_data_point], X[-test_data_point:], hat_y[-test_data_point:]

theta, X = idltraining.initialize_theta_2(X_train.T, y_train, 300, starting_lambda=None,
                                          tol_fenchtel=1e-5, verbose=False,
                                          random_state=0)
IDL = idl.IDLModel(hidden_variables=300, dual_learning_rate=100, tol_fenchel=1e-5,
                   random_state=0, verbosity=True, inner_tol=1e-3, initialization_theta=(theta.copy(), X.copy()),)
y_pred = IDL.predict(X_test)
results_before = (np.sqrt(mean_squared_error(y_test, y_pred[0])))
IDL.fit(X_train, y_train, rounds_number=200, verbose=True, type_of_training="two_loops", eval_set=(X_test, y_test))
y_test_predict = IDL.predict(X_test)[0]
results = (np.sqrt(mean_squared_error(y_test, y_test_predict)))



baseline = LinearRegression()
baseline.fit(X_train, y_train)
# model evaluation for testing set
y_test_predict = baseline.predict(X_test)
rmse_linear = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# print("RMSE IDL : ", rmse_IDL, " vs RMSE Linear Regression : ", rmse_linear)