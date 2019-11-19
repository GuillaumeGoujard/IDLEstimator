import sys, os, numpy as np

# The following file conducts unit tests for the function project_to_S_theta in utilities.ConvexProjection
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/path/to/application/app/folder')

sys.path.append("../")

from utilities import GradientDescents as grad


def initial_test():
    h = 5
    p = 4
    n = 7
    A = np.random.rand(p, h)
    D = np.random.rand(h, h)
    c = np.random.rand(p, 1)
    B = np.random.rand(p, n)
    E = np.random.rand(h, n)
    f = np.random.rand(h, 1)
    Lambda = np.diag(np.random.rand(h))

    m = 6

    theta = {"A": A, "D": D, "Lambda": Lambda, "c": c, "B": B, "E": E, "f": f, "m": m}


    X = np.random.rand(h, m)
    Y = np.random.rand(p, m)

    U = np.random.rand(n, m)
    #grad_X = grad.gradient_descent_x(theta, X, U)

    #grad_theta = grad.gradient_descent_theta(theta, X, U, Y)
    new_theta = grad.update_theta(theta, X, U, Y)
    a = 2



if __name__ == '__main__':
    initial_test()
