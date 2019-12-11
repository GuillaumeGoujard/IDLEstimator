import IDL as idl
import numpy as np
import matplotlib.pyplot as plt

"""
Let us try to make it learn a very simple model :

Y = X + epsilon
"""
number_of_data_points = 1000
test_data_point = int(0.1*number_of_data_points)
plot_training =False

def create_regressive_model(n_samples, a=1, noise_std = 0.01):
    X = np.random.random_sample(n_samples)
    return model(X, a=a, noise_std=noise_std)

def model(X, a=2, noise_std =0.01):
    n_samples = X.shape[0]
    epsilons = np.random.normal(0, noise_std, n_samples)
    return X.reshape((-1, 1)), a * X + epsilons


from sklearn.metrics import mean_squared_error, r2_score

X, y = create_regressive_model(number_of_data_points, a=1, noise_std=0.1)
X_train, y_train, X_test, y_test = X[:-test_data_point], y[:-test_data_point], X[-test_data_point:], y[-test_data_point:]

if plot_training:
    plt.scatter(X_train, y_train)
    plt.grid(True)
    plt.title("Training Set")
    plt.show()


"""
IDL model
"""
# hidden_variables = 4
# dual_learning_rate = 0.1
# tol_fenchtel = 0.01
# IDL = idl.IDLModel(hidden_variables=hidden_variables, dual_learning_rate=dual_learning_rate, tol_fenchtel=tol_fenchtel,
#                    random_state=0, verbosity=True)
# IDL.fit(X_train, y_train, rounds_number=50, verbose=True, type_of_training="two_loops")
#
# """
# Let us try to predict !
# """
# y_pred = IDL.predict(X_test)
# X_test_model = X_test.reshape(1, -1).copy()
# X_test_model.sort()
# plt.plot(X_test_model[0], model(X_test_model, a=1, noise_std=0)[1][0], color="red", label="denoised model")
# plt.scatter(X_test, y_test, label="y_test")
# plt.scatter(X_test, y_pred, label="predictions")
# plt.grid(True)
# plt.legend()
# plt.title("Predictions over the test set")
# plt.show()


"""
Let us try to make it learn a quadratic  model :

Y = X^2 + epsilon
"""

def create_quadratic_model(n_samples, noise_std = 0.01):
    X = np.random.random_sample(n_samples)*np.pi*2
    return quadratic_model(X, noise_std=noise_std)

def quadratic_model(X, noise_std =0.01):
    n_samples = X.shape[0]
    epsilons = np.random.normal(0, noise_std, n_samples)
    return X.reshape((-1, 1)), np.sin(X) #np.power(X,2) + epsilons

X, y = create_quadratic_model(number_of_data_points, noise_std=1)
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
dual_learning_rate = 1
tol_fenchtel = 0.001


IDL = idl.IDLModel(hidden_variables=hidden_variables, dual_learning_rate=dual_learning_rate, tol_fenchtel=tol_fenchtel,
                   random_state=0, verbosity=True)
IDL.fit(X_train, y_train, rounds_number=100, verbose=True, type_of_training="two_loops")

# results = {}
# plots = {}
# from sklearn.metrics import mean_squared_error, r2_score
# hidden_variables = [10, 100, 250]
# hidden_variables = [100]
# for h in hidden_variables:
#     print(h)
#     IDL = idl.IDLModel(hidden_variables=h, dual_learning_rate=dual_learning_rate, tol_fenchtel=tol_fenchtel,
#                        random_state=0, verbosity=True, inner_tol=1e-3)
#     IDL.fit(X_train, y_train, rounds_number=400, verbose=True, type_of_training="two_loops")
#     y_test_predict = IDL.predict(X_test)[0]
#     y_pred = IDL.predict(X_test)
#     results[h] = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
#     plots[h] = IDL.evals_result
#     X_test_model = X_test.reshape(1, -1).copy()
#     X_test_model.sort()
#     plt.plot(X_test_model[0], quadratic_model(X_test_model, noise_std=0)[1][0], color="red", label="denoised model")
#     plt.scatter(X_test, y_test, label="y_test")
#     plt.scatter(X_test, y_pred, label="predictions")
#     plt.grid(True)
#     plt.legend()
#     plt.title("Predictions over the test set h = " + str(h))
#     plt.show()



"""
Let us try to predict !
"""
y_pred = IDL.predict(X_test)
X_test_model = X_test.reshape(1, -1).copy()
X_test_model.sort()
plt.plot(X_test_model[0], quadratic_model(X_test_model, noise_std=0)[1][0], color="red", label="denoised model")
plt.scatter(X_test, y_test, label="y_test")
plt.scatter(X_test, y_pred, label="predictions")
plt.grid(True)
plt.legend()
plt.title("Predictions over the test set")
plt.show()

plt.clf()
plt.scatter(X_train, IDL.predict(X_train))
plt.scatter(X_train, y_train)
plt.grid(True)
plt.show()

