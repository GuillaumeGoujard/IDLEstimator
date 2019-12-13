import IDL as idl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import training as idltraining

"""
Let us try to make it learn a very simple model :

Y = X + epsilon
"""
number_of_data_points = 1000
test_data_point = int(0.1*number_of_data_points)
plot_training = True

def linear_example():
    def create_regressive_model(n_samples, a=1, noise_std = 0.01):
        X = np.random.random_sample(n_samples)
        return model(X, a=a, noise_std=noise_std)

    def model(X, a=2, noise_std =0.01):
        n_samples = X.shape[0]
        epsilons = np.random.normal(0, noise_std, n_samples)
        return X.reshape((-1, 1)), a * X + epsilons


    X, y = create_regressive_model(number_of_data_points, a=1, noise_std=0.5)
    X_train, y_train, X_test, y_test = X[:-test_data_point], y[:-test_data_point], X[-test_data_point:], y[-test_data_point:]

    if plot_training:
        plt.scatter(X_train, y_train)
        plt.grid(True)
        plt.title("Training Set")
        plt.show()


    """
    IDL model
    """
    hidden_variables = 1
    dual_learning_rate = 1
    tol_fenchtel = 0.01
    IDL = idl.IDLModel(hidden_variables=hidden_variables, dual_learning_rate=dual_learning_rate, tol_fenchel=tol_fenchtel,
                       random_state=0, verbosity=True, early_stopping=True, starting_lambda=None)
    IDL.fit(X_train, y_train, rounds_number=50, verbose=True, type_of_training="two_loops", eval_set=(X_test, y_test))


    """
    Let us try to predict !
    """
    y_pred = IDL.predict(X_test)
    X_test_model = X_test.reshape(1, -1).copy()
    X_test_model.sort()
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X_train, y_train)  # perform linear regression
    Y_pred_regr = linear_regressor.predict(X_test_model[0].reshape(-1,1))  # make predictions

    plt.plot(X_test_model[0], model(X_test_model, a=1, noise_std=0)[1][0], color="red", label="denoised model")
    plt.plot(X_test_model[0], Y_pred_regr, label="regression model")
    plt.scatter(X_test, y_test, label="y_test")
    plt.scatter(X_test, y_pred, label="predictions")
    plt.grid(True)
    plt.legend()
    plt.title("Predictions over the test set")
    plt.show()



"""
Let us try to make it learn a sinus model :

Y = sin(X) + epsilon
"""

def sin_example():
    def create_sin_model(n_samples, noise_std = 0.01):
        X = np.random.random_sample(n_samples)*np.pi*2
        return quadratic_model(X, noise_std=noise_std)

    def quadratic_model(X, noise_std =0.01):
        n_samples = X.shape[0]
        epsilons = np.random.normal(0, noise_std, n_samples)
        return X.reshape((-1, 1)), np.sin(X) + epsilons

    X, y = create_sin_model(number_of_data_points, noise_std=0.1)
    X_train, y_train, X_test, y_test = X[:-test_data_point], y[:-test_data_point], X[-test_data_point:], y[-test_data_point:]

    if True:
        plt.scatter(X_train, y_train)
        plt.grid(True)
        plt.title("Training Set")
        plt.show()


    """
    IDL model
    """

    hidden_variables = 100
    dual_learning_rate = 1000
    tol_fenchtel = 0.0001
    starting_lambda = tol_fenchtel
    import training as idltraining
    U = X_train.T.copy()
    theta, X = idltraining.initialize_theta_2(U, y_train, hidden_variables, starting_lambda=None,
                                            tol_fenchtel=tol_fenchtel, verbose=True,
                                            random_state=0)


    """
    First example : No tuning
    """
    lambda0 = [ 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 ]
    dual_learning_rates = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 ]
    tol_fenchtels =[ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 ]
    results = {}
    for d in tol_fenchtels:
        print(d)

        IDL = idl.IDLModel(hidden_variables=hidden_variables, dual_learning_rate=1e-6, tol_fenchel=d,
                           initialization_theta=(theta.copy(), X.copy()), starting_lambda=None,
                           random_state=0, verbosity=False, early_stopping=True)
        IDL.fit(X_train, y_train, rounds_number=200, verbose=False, type_of_training="two_loops",
                eval_set=(X_test, y_test))
        y_pred = IDL.predict(X_test)
        results[d] = np.sqrt(mean_squared_error(y_test, y_pred[0]))

    plt.plot(np.log(dual_learning_rates), list(results.values()))
    plt.show()

