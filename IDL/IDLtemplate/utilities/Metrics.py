import numpy as np
import matplotlib.pyplot as plt

def fenchtel_div(X, Y):
    Z = 0.5*(np.multiply(X, X) + np.multiply(np.maximum(0, Y), np.maximum(0, Y))) - np.multiply(X, Y)
    return np.mean(Z, axis=1)


def loss(U, y, theta, X):
    test_satisfies_contraints(y, U, theta, X)
    lambda_ = np.diag(theta["Lambda"])
    m = theta["m"]
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    y_fenchtel = D@X + E@U + f@np.ones((1,m))
    return L2Loss(U, y, theta, X) + lambda_@fenchtel_div(X, y_fenchtel)


def test_satisfies_contraints(y, U, theta, X):
    pass


def L2Loss(U, y, theta, X, evaluating=False):
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    m = theta["m"] if evaluating is False else U.shape[1]
    M = A@X + B@U + c@np.ones((1,m)) - y
    return (1/(2*m))*np.linalg.norm(M, ord="fro")**2


def fenchtel_error(theta, X, U, lambda_=None):
    m = theta["m"]
    A, B, c, D, E, f, Lambda = theta["A"], theta["B"], theta["c"], theta["D"], theta["E"], theta["f"], theta["Lambda"]
    y_fenchtel = D @ X + E @ U + f @ np.ones((1, m))
    if lambda_ is None:
        lambda_ = np.diag(theta["Lambda"])
    return np.float(lambda_ @ fenchtel_div(X, y_fenchtel))


def how_far_from_RELU(U, X, theta):
    return np.linalg.norm(X - np.maximum(0, theta["D"] @ X + theta["E"] @ U + theta["f"] @ np.ones((1, theta["m"]))),
                   ord="fro")


def plot_training_errors(training_errors):
    L2_loss = np.array(training_errors)

    fig, ax1 = plt.subplots(figsize=(12,8))

    color = 'tab:red'
    ax1.set_xlabel('rounds')
    ax1.set_ylabel('General Loss', color=color)
    ax1.plot(L2_loss[:, 0][1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('L2 loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(L2_loss[:, 1][1:], label="L2 loss on train set")
    if L2_loss.shape[1] == 5 :
        ax2.plot(L2_loss[:, 4][1:], label="L2 loss on eval set")
    ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Training Errors in function of round")
    plt.grid(True)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('rounds')
    color = 'tab:red'
    ax1.set_ylabel('1 norm of lambda vector', color=color)
    ax1.plot(L2_loss[:, 2][1:], color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('gap between X and (X...)_+', color=color)  # we already handled the x-label with ax1
    ax2.plot(L2_loss[:, 3][1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Fenchel Errors in function of round")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()
