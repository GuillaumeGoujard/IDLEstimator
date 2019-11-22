import IDL as idl
import numpy as np
import matplotlib.pyplot as plt

"""
Let us try to make it learn a very simple model :

Y = 2*X + epsilon
"""

def create_regressive_model(n_samples, a=2, noise_std = 0.01):
    X = np.random.random_sample(n_samples)
    epsilons = np.random.normal(0, noise_std, n_samples)
    return X.reshape((-1,1)), a*X + epsilons

X, y = create_regressive_model(100, a=2, noise_std=0.01)
X_train, y_train, X_test, y_test = X[:90], y[:90], X[-10:], y[-10:]

plt.scatter(X_train, y_train)
plt.grid(True)
plt.title("Training Set")
plt.show()

print("X_train has shape : ", X_train.shape)
print("y_train has shape : ", y_train.shape)

IDL = idl.IDLModel(hidden_features=5)
IDL.fit(X, y, rounds_number=100, early_stopping_rounds=10, verbose=True)

"""
Test of the training model
"""
U = X.T.copy()
Y_hat = IDL.theta["A"]@IDL.training_X + IDL.theta["B"]@U + IDL.theta["c"]@np.ones((1,IDL.theta["m"]))
plt.scatter(X, Y_hat)
plt.grid(True)
plt.title("Prediction over the training set... ")
plt.show()

"""
Let us try to predict !
"""
y_pred = IDL.predict(X_test)
plt.scatter(X_test, y_test, label="test set")
plt.scatter(X_test, y_pred, label="predictions")
plt.grid(True)
plt.title("Predictions over the test set !")
plt.show()



