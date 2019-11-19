import sys, os, numpy as np

# The following file conducts unit tests for the function project_to_S_theta in utilities.ConvexProjection
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/path/to/application/app/folder')

sys.path.append("../")

from utilities import ConvexProjection as conv


def initial_test():
    h = 5
    p = 4

    A = np.random.rand(p, h)
    D = np.random.rand(h, h)

    Lambda = np.diag(np.random.rand(h))
    theta = {"A": A, "D": D, "Lambda": Lambda}

    result = conv.project_to_S_theta(theta)
    newresult = conv.project_to_S_theta(theta)

    print(result)
    a = 2



if __name__ == '__main__':
    print(dir(conv))
    initial_test()
