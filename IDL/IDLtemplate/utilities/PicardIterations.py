import numpy as np
import matplotlib.pyplot as plt

def picard_iterations(X, D, delta, k_iterations=100, tol=1e-10):
    i = 0
    while i < k_iterations:
        X_ = np.maximum(0, D@X + delta)
        if np.linalg.norm(X - X_, ord="fro") < tol:
            return X
        # print(np.linalg.norm(X - X_, ord="fro"))
        X = X_
        i += 1
    return X

def plot_before_after_picard(U, y, theta, X, k):
    plt.scatter(U[0, :], y, label="y_train")
    plt.scatter(U[0, :],
                theta["A"] @ X + theta["B"] @ U + theta["c"] @ np.ones((1, theta["m"])),
                label="Prediction over X_train, before enforcing RELU constraint")
    print(np.linalg.norm(X - np.maximum(0, theta["D"] @ X + theta["E"] @ U + theta["f"] @ np.ones((1, theta["m"]))),
                         ord="fro"))
    Xpic = picard_iterations(X, theta["D"], theta["E"] @ U + theta["f"] @ np.ones((1, theta["m"])),
                                k_iterations=1000)
    plt.scatter(U[0, :], theta["A"] @ Xpic + theta["B"] @ U + theta["c"] @ np.ones((1, theta["m"])),
                label="Prediction over X_train, after enforcing RELU constraint")
    plt.legend()
    plt.title("Fit of the training set round : " + str(k))
    plt.grid(True)
    plt.show()

