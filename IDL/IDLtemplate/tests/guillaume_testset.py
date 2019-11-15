import sys
sys.path.extend(['/Users/guillaumegoujard/Desktop/Fall Semester/EE227B_project/IDL/IDLtemplate'])
from utilities import Metrics, DualAscents
import numpy as np

X = np.array([[-1,2,1],[3,4,1]])
Y = np.array([[-2,-2,1],[-3,-3,2]])
Z = np.multiply(X,X) + np.multiply(np.maximum(0,Y), np.maximum(0,Y)) - np.multiply(X,Y)

Z = Metrics.fenchtel_div(X,Y)

"""
Toy example 1
"""
x = np.array([1,2])
A = np.array([[2,0],[1,2]])
B = np.array([[1,0],[0,1]])
u = np.array([-0.5,-0.5])
c = np.array([-0.5,-2.5])

x == A@x + B@u + c

# Z = Metrics.fenchtel_div(x, A@x + B@u + c)
# Z = Metrics.fenchtel_div(x, A@np.array([1,1.7]) + B@u + c)

"""
Toy example 2
"""
p = 3
y =  np.random.normal(0, 1, (p))

n = 10
N = 100
U = np.random.normal(0, 1, (n, N))

h = 5
x = np.random.normal(0, 1, (h, N))

A = np.ones((p,h))
B = np.ones((p,n))
c = np.ones((p))
D = np.ones((h,h))
E = np.ones((h,n))
f = np.ones((h))

lambda_dual = np.ones((h))
Lambda = np.diag(lambda_dual)
theta = [A, B, c, D, E, f, Lambda]

"""
Test the dual ascent
"""
lambda_dual = DualAscents.update_dual(theta, x, U)
# m = x.shape[1]
# F = Metrics.fenchtel_div(x, D@x + E@U + f.reshape((f.shape[0],1))@np.ones((1,m)))
# indicator = np.zeros(F.shape)
# indicator[F > 0.5] = 1
# dlambda = 0.2*np.multiply(F, indicator)
