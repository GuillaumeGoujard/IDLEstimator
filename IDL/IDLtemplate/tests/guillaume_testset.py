import sys
sys.path.extend(['/Users/guillaumegoujard/Desktop/Fall Semester/EE227B_project/IDL/IDLtemplate'])
from utilities import Metrics, DualAscents
from utilities import GradientDescents as gd
from utilities import ConvexProjection as cp
from utilities import Metrics as me
from utilities import DualAscents as da
import numpy as np

# X = np.array([[-1,2,1],[3,4,1]])
# Y = np.array([[-2,-2,1],[-3,-3,2]])
# Z = np.multiply(X,X) + np.multiply(np.maximum(0,Y), np.maximum(0,Y)) - np.multiply(X,Y)
#
# Z = Metrics.fenchtel_div(X,Y)
#
# """
# Toy example 1
# """
# x = np.array([1,2])
# A = np.array([[2,0],[1,2]])
# B = np.array([[1,0],[0,1]])
# u = np.array([-0.5,-0.5])
# c = np.array([-0.5,-2.5])
#
# x == A@x + B@u + c
#
# # Z = Metrics.fenchtel_div(x, A@x + B@u + c)
# # Z = Metrics.fenchtel_div(x, A@np.array([1,1.7]) + B@u + c)
#
# """
# Toy example 2
# """
# p = 3
# y =  np.random.normal(0, 1, (p))
#
# n = 10
# N = 100
# U = np.random.normal(0, 1, (n, N))
#
# h = 5
# x = np.random.normal(0, 1, (h, N))
#
# A = np.ones((p,h))
# B = np.ones((p,n))
# c = np.ones((p))
# D = np.ones((h,h))
# E = np.ones((h,n))
# f = np.ones((h))
#
# lambda_dual = np.ones((h))
# Lambda = np.diag(lambda_dual)
# theta = [A, B, c, D, E, f, Lambda]
#
# """
# Test the dual ascent
# """
# lambda_dual = DualAscents.update_dual(theta, x, U)
# # m = x.shape[1]
# # F = Metrics.fenchtel_div(x, D@x + E@U + f.reshape((f.shape[0],1))@np.ones((1,m)))
# # indicator = np.zeros(F.shape)
# # indicator[F > 0.5] = 1
# # dlambda = 0.2*np.multiply(F, indicator)
#

"""
Initialize
"""

X = np.array([[1,2, 3]])
y = np.array([[1,2,3]])

U = X.copy()

"""
Initialization of the theta vector
"""
n, m = U.shape
p, m = y.shape
h = 2
A = np.random.normal(0, 1, (p, h))
B = np.random.normal(0, 1,(p, n))
c = np.random.normal(0, 1,(p, 1))
D = np.random.normal(0, 1,(h, h))
E = np.random.normal(0, 1,(h, n))
f = np.random.normal(0, 1,(h, 1))
X = np.random.normal(0, 1, (h, m))

lambda_dual = np.ones((h))*0.1
Lambda = np.diag(lambda_dual)
theta = {"A": A, "B": B, "c": c, "D": D, "E": E, "f": f, "Lambda": Lambda, "m": m}

A@X+ B@U + c@np.ones((1,m)) - y
#

# L = me.loss(y, U, theta, X)
#
# theta = gd.update_theta(theta, X, U, y)
# theta = cp.project_to_S_theta(theta)
#
# X = np.maximum(0, X - gd.alpha_x(theta)*gd.gradient_descent_x(theta, X,  U))
# lambda_dual = np.diag(da.update_dual(theta, X, U))
#

for k in range(100):
    L = me.loss(y, U, theta, X)
    print(k, " loss is ", L)
    theta = gd.update_theta(theta, X, U, y)
    # theta = theta - gd.alpha_theta(X, U)@gd.gradient_descent_theta(X, theta, U, y)
    theta = cp.project_to_S_theta(theta)
    X = np.maximum(0, X - gd.alpha_x(theta) * gd.gradient_descent_x(theta, X,  U))
    theta["Lambda"] = da.update_dual(theta, X, U)

theta["A"]@X + theta["B"]@U + theta["c"]@np.ones((1,3))