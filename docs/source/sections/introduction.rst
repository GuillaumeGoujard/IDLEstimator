Introduction
============

This package can be used to fit an Implicit Deep Learning (IDL) model for regression
and classification purpose.

The IDL.fit function estimates a vector of parameters by applying successively
gradient descents (see more at :ref:`Gradient Descents`) and dual ascent
(see more at :ref:`Dual Ascents`).

.. _Implicit Deep Learning:
Implicit Deep Learning
*************************

Given an input :math:`u \in \mathbb{R}^n`, where n denotes the number of features,
we define the implicit deep learning prediction rule :math:`\hat{y}(u) \in \mathbb{R}^n` with ReLU activation

.. math::
    \begin{align}
        \hat{y}(u) &= Ax + Bu + c \\
        x &= (Dx + Eu + f)_+,
    \end{align}
    :label: eq_1

where :math:`(.)_+ := \text{max}(0,.)` is ReLU activation, :math:`x \in \mathbb{R}^h` is called the hidden variable
(h is the number of hidden features), :math:`\Theta := (A,B,c,D,E,f)` are matrices and vectors of appropriate size, they define the
parameters of the model. The hidden variable :math:`x` is implicit in the sense that there is in general no analytical
formula for it, this is different from classic deep learning for which, given the model parameters, the hidden
variables can be computed explicitly via propagation through the network.

Notation and definitions
*************************
We denote :math:`\Vert . \Vert` the eucledian norm, :math:`\Vert . \Vert_2` the corresponding norm (i.e. the spectral norm) and
:math:`\Vert . \Vert_F` the Frobenius norm. :math:`\mathbb{R}_+^n` denotes the positive orthant of the vector space :math:`\mathbb{R}^n, \mathbb{S}^n`
the set of real symmetric matrices of size :math:`n` and :math:`\mathbb{S}_+^n` the cone of positive semi-definite matrices of size :math:`n`. The transpose of a matrix or
vector is denoted :math:`.^T` and elementwise product is denoted :math:`\odot`. Given a differentiable function :math:`f` from :math:`\mathbb{R}^{n \times p}` to :math:`\mathbb{R}`
we define the scalar by matrix partial derivative in denominator layout convention as

.. math::
    \frac{\partial f}{\partial A} = \nabla_A f = \begin{bmatrix}
            \frac{\partial f}{\partial A_{1,1}} & \cdots & \frac{\partial f}{\partial A_{1,p}} \\
            \vdots & \ddots & \vdots \\
            \frac{\partial f}{\partial A_{n,1}} & \cdots & \frac{\partial f}{\partial A_{n,p}}
        \end{bmatrix}
        \in \mathbb{R}^{n \times p}.

We say that a function :math:`(x,y) \rightarrow f(x,y)` with seperable domain of definition :math:`\mathcal{X} \times \mathcal{Y}` is bi-convex in :math:`(x,y)`,
if for all :math:`x \in \mathcal{X}`, the function :math:`y \rightarrow f(x,y)` is convex and for all :math:`y \in \mathcal{Y}` the function :math:`x \rightarrow f(x,y)` is convex.
We say that a function is smooth if it is differentiable and its gradient is Lipschitz continious. We say that :math:`f` is bi-smooth if it is smooth in :math:`x` given :math:`y` and
vice-versa. An example of bi-smooth and bi-convex function is :math:`(x,A) \rightarrow x^TAx, A \in \mathbb{S}_+^n`.

Well-posedness
*************************
We say that matrix :math:`D` is well-posed for :eq:`eq_1` if there exists a unique solution :math:`x = (Dx + \delta)_+ \forall \delta \in \mathbb{R}^h`.
Using the fact that ReLU is 1-Lipschitz we have for :math:`x_1,x_2 \in \mathbb{R}^h`

.. math::
    \Vert (Dx_1 + \delta)_+ - (Dx_2 + \delta)_+ \Vert \leq \Vert D(x_1 -x_2) \Vert \leq \Vert D \Vert_2 \Vert x_1 -x_2 \Vert.

If :math:`\Vert D \Vert_2 < 1` we have that the map :math:`x \rightarrow (Dx + \delta)_+` is a strict contraction. In that case, Banach's contraction
mapping theorem applies, showing that the equation :math:`x = (Dx + \delta)_+` has a unique solution. In that case, a solution :math:`x` can be computed via the
Picard iterations

.. math::
    x^{k+1} = (Dx + \delta), k = 1,2, \cdots.

Note that :math:`\Vert D \Vert_2 < 1` is only a sufficient condition for well-posedness. Nevertheless this is the only condition
we will consider in this article.

Setup
******
TODO

The package is compatible with Python version 3 or higher only.
The user is expected to have installed cvxpy before running the package.
Go to ... for more information.

1. Switch to a proper directory and then type:

::

    git clone + https://github.com/...
