import numpy as np

"""
Let's say we have the following vector of predictor : \hat{y} with p variable p = 3
"""
p = 3
y =  np.random.normal(0, 1, (p))

"""
Let's say we have a vector of N=100 data-points with n=10 features :
"""
n = 10
N = 100
U = np.random.normal(0, 1, (n, N))

"""
Let's say the number of hidden features is 5
"""
h = 5
x = np.random.normal(0, 1, (h))

"""
It implies for theta :

    - A \in R^{p,h}
    - B \in R^{p,n}
    - c \in R^{p}
    - D \in R^{h,h}
    - E \in R^{h,n}
    - f \in R^{h}
"""
A = np.ones((p,h))
B = np.ones((p,n))
c = np.ones((p))
D = np.ones((h,h))
E = np.ones((h,n))
f = np.ones((h))

"""
These operations are licit
"""
hat_y = A@x + B@U[:,1] + c
x = np.maximum(0, D@x + E@U[:,1] + f)