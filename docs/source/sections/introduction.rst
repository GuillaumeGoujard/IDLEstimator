Introduction
============

numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq.{number}"

This package can be used to fit an Implicit Deep Learning (IDL) model for regression
and classification purpose.

The IDL.fit function estimates a vector of parameters by applying successively
gradient descents (see more at :ref:`Gradient Descents`) and dual ascent
(see more at :ref:`Dual Ascents`).

.. _Implicit Deep Learning:
1.1 Implicit Deep Learning
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
variables can be computed explicitly via propagation trough the network.

1.2 Notation and definitions
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

2 Well-posedness
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

3.1 Problem formulation
*************************
Let us consider the input and output data matrices :math:`U = [u_1, \cdots, u_m] \in \mathbb{R}^{n \times m},Y = [y_1, \cdots, y_m] \in \mathbb{R}^{p \times m}`
with :math:`m` being the number of datapoints - the corresponding optimization regression problem (i.e. with
squared error loss) reads

.. math::
    \begin{align}
        \min_{\color{blue}{X}, \color{blue}{\Theta}} \quad &\mathcal{L}(Y,[\color{blue}{\Theta},\color{blue}{X}]) := \frac{1}{2m} \Vert \color{blue}{AX} + \color{blue}{B}U + \color{blue}{c}1_m^T - Y \Vert_F^2 \\
        \text{s.t.} \quad &\color{blue}{X}=(\color{blue}{DX} + \color{blue}{E}U + \color{blue}{f}1_m^T)_+ \\
                    \quad &\Vert \color{blue}{D} \Vert_2 < 1,
    \end{align}
    :label: eq_2

where :math:`1_m` is a column vector of size :math:`m` consisting of ones. For clarity we have highlighted in \color{blue}{blue} the optimization variables.
The non-convexity of this problem arises from the nonlinear implicit constraint and
the matrix product terms :math:`AX` and :math:`DX`. In practice we replace the constraint :math:`\Vert D \Vert_2 < 1`
by the closed convex form :math:`\Vert D \Vert_2 \leq 1 - \epsilon`, where :math:`\epsilon > 0` is small.

3.2 Fenchel Divergence Lagrangian Relaxation
*************************
Using Fenchel-Young inequality, it can be shown that the equation :math:`x = (Dx + Eu + f)_+` is equivalent to

.. math::
    \begin{cases}
        \mathcal{F}(x,Ax + Bu + c) = 0 \\
        x \geq 0
    \end{cases}
    :label: eq_3

with the Fenchel Divergence :math:`\mathcal{F}` defined by

.. math::
    \mathcal{F}(x_1,x_2) := \frac{1}{2} x_1 \odot x_1 + \frac{1}{2} (x_2)_+ \odot (x_2)_+ - x_1 \odot x_2.

We use the term divergence because by construction :math:`\mathcal{F}(x_1,x_2) \geq 0 \forall x_1,x_2 \in \mathbb{R}_+^h \times \mathbb{R}^h`.
Given :math:`X = [x_1, \cdots, x_m]` and :math:`Z = [z_1, \cdots, z_m]` we write

.. math::
    \mathcal{F}(X,Z) := \frac{1}{m} \sum_{i=1}^m \mathcal{F}(x_i,z_i).

A Lagrangian relaxation approach to solving :eq:`eq_2` problem using the implicit constraint formulation :eq:`eq_3` consists in
solving given a dual variable :math:`\lambda \in \mathcal{R}_+^h`

.. math::
    \begin{align}
        \min_{\color{blue}{X} \geq 0, \color{blue}{\Theta}} \quad &\mathcal{L}(Y,[\color{blue}{\Theta},\color{blue}{X}]) + \lambda^T \mathcal{F}(\color{blue}{X},\color{blue}{DX} + \color{blue}{E}U + \color{blue}{f}1_m^T) \\
        \text{s.t.} \quad &\Vert \color{blue}{D} \Vert_2 < 1,
    \end{align}
    :label: eq_4

This problem is bi-smooth in :math:`[\Theta,X]`, but it is not convex or bi-convex. Nevertheless we can make it bi-convex
with extra conditions on :math:`\Theta` as shown in the next section.

3.3 Linear matrix inequality parameter constraints for bi-convexity
*************************
Let us define :math:`\Lambda = diag(\lambda) \in \mathbb{S}_+^h`

Theorem 1. *Problem* :eq:`eq_4` *is bi-convex in* :math:`[\Theta,X]` *if we impose one of the two following feasible linear matrix inequalities (LMI)*

.. math::
    \Lambda - (\Lambda D + D^T \Lambda) \in \mathbb{S}_+^h \\
    :label: eq_5

.. math::
    \Lambda + A^TA - (\Lambda D + D^T \Lambda) \in \mathbb{S}_+^h \\
    :label: eq_6

*Proof.* The loss term :math:`\mathcal{L}(Y,[\Theta,X])` is already bi-convex in :math:`(\Theta,X)`, but it is not the case for the Fenchel
Divergence term :math:`\lambda^T \mathcal{F}(X,DX + EU + f1_m^T)`, which is not convex in :math:`X` in the general case. A sufficient
condition for this term to be convex in :math:`X` given :math:`\Theta` is for the following function

.. math::
    x \rightarrow \lambda^T(\frac{1}{2} x \odot x - x \odot Dx) = \frac{1}{2}x^T(\Lambda - (\Lambda D + D^T \Lambda))x,

to be convex. This term is convex in :math:`x` if the LMI :eq:`eq_5` is satisfied. Now the second LMI similarly arises by
leveraging the fact that we can also use the term in the loss to make the objective convex in :math:`x`. Indeed the
objective function of :eq:`eq_2` is convex in :math:`x` if

.. math::
    x \rightarrow \frac{1}{2}x^TA^TAx + \frac{1}{2}x^T(\Lambda - (\Lambda D + D^T \Lambda))x,

is convex, which corresponds to LMI :eq:`eq_6`. It might not be obvious that :eq:`eq_6` is actually an LMI, but using
Schur complement we can prove it is equivalent to

.. math::
    - \begin{bmatrix}
            I_p & A \\
            A^T & \Lambda D + D^T \Lambda - \Lambda
        \end{bmatrix}
        \in \mathbb{S}_+^{p + h}.


If D satisfies :eq:`eq_5` then it satisfies :eq:`eq_6`. We imediately have that :math:`D = \delta I_n` with :math:`\delta \leq \frac{1}{2}` satisfies :eq:`eq_5`
(and :math:`\Vert D \Vert_2 \leq 1 - \epsilon`). Which proves that both LMIs are feasible.

From this proof, the problem formulation reads

.. math::
    \begin{align}
        \min_{\color{blue}{X} \geq 0, \color{blue}{\Theta}} \quad &\mathcal{L}(Y,[\color{blue}{\Theta},\color{blue}{X}]) + \lambda^T \mathcal{F}(\color{blue}{X},\color{blue}{DX} + \color{blue}{E}U + \color{blue}{f}1_m^T) \\
        \text{s.t.} \quad &\Vert \color{blue}{D} \Vert_2 \leq 1 - \epsilon \\
                    \quad &\Lambda + \color{blue}{A}^T\color{blue}{A} - (\Lambda \color{blue}{D} + \color{blue}{D}^T \Lambda) \in \mathbb{S}_+^h,
    \end{align}
    :label: eq_7

this problem is well-posed - feasible solutions exist - and bi-smooth.

3.4 Block coordinate descent and first order methods
*************************
As problem :eq:`eq_7` is bi-convex, a natural strategy is the use of block coordinate descent (BCD): alternating
optimization between :math:`\Theta`and :math:`X`. BCD corresponds to the following algorithm,

**for** :math:`k = 1, 2, \cdots` **do**

.. math::
    \begin{align}
        \Theta^k \in \text{argmin}_{\Theta} &\frac{1}{2m} \Vert AX^{k-1} + BU + c1_m^T - Y \Vert_F^2 + \lambda^T \mathcal{F}(X^{k-1},DX^{k-1} + EU + f1_m^T) \\
            &\quad \Lambda + A^TA - (\Lambda D + D^T \Lambda) \in \mathbb{S}_+^h, \\
            &\quad \Vert D \Vert_2 \leq 1 - \epsilon \\
        X^k \in \text{argmin}_{X \geq 0} &\frac{1}{2m}\Vert A^kX + B^kU + c^k1_m^T - Y \Vert_F^2 + \lambda^T \mathcal{F}(X,D^kX + E^kU + f^k1_m^T)
    \end{align}
**end**

In practice such updates might be to heavy computationally as the number of datapoints :math:`m` increase, or
as the model size increases (i.e. :math:`h`, :math:`n` or :math:`p`). Instead we propose to do block coordinate projected gradient
updates. This method is also considered to be better at avoiding local minima. Let us denote

.. math::
    \mathcal{G}(\Theta,X) := \mathcal{L}(Y,[\Theta,X]) + \lambda^T \mathcal{F}(X,DX + EU + f1_m^T)

In the remainder of this section we derive the gradients :math:`\nabla_{\Theta} \mathcal{G}(\Theta,X), \nabla_X \mathcal{G}(\Theta,X)` and corresponding 'optimal'
step-sizes using the Lipschitz coefficients of the gradients- which is the advantage of having a bi-smooth
optimization problem. Note that the objective :math:`\mathcal{G}`, given :math:`X` is separable in :math:`\Theta_1 := (A,B,c)` and :math:`\Theta_2 := (D,E,f)`.
Using scalar by matrix calculus

.. math::
	\begin{cases}
		\nabla_A \mathcal{G}(\Theta,X) = \Omega(A,B,c)X^T \in \mathbb{R}^{p \times h} \\
		\nabla_B \mathcal{G}(\Theta,X) = \Omega(A,B,c)U^T \in \mathbb{R}^{p \times n} \\
        \nabla_c \mathcal{G}(\Theta,X) = \Omega(A,B,c)1_m \in \mathbb{R}^p
	\end{cases},

with :math:`\Omega(A,B,c) := \frac{1}{m}(AX + BU + c1_m^T - Y) \in \mathbb{R}^{p \times m}`. Hence we can show that a Lipschitz constant for the
gradient is given by

.. math::
    L_{\Theta_1}(X) := \frac{1}{m} \max(m,\Vert X \Vert_2^2,\Vert U \Vert_2^2,\Vert XU^T \Vert_2),

and the 'optimal' step-size for gradient descent is then simply given by

.. math::
    \alpha_{\Theta_1}(X) := \frac{1}{L_{\Theta}(X)}.

Regarding the gradient with respect to :math:`\Theta_2`, we have

.. math::
	\begin{cases}
		\nabla_D \mathcal{G}(\Theta,X) = \Omega(D,E,f,\Lambda)X^T \in \mathbb{R}^{h \times h} \\
		\nabla_E \mathcal{G}(\Theta,X) = \Omega(D,E,f,\Lambda)U^T \in \mathbb{R}^{h \times n} \\
        \nabla_f \mathcal{G}(\Theta,X) = \Omega(D,E,f,\Lambda)1_m \in \mathbb{R}^h
	\end{cases},

with :math:`\Omega(D,E,f,\Lambda) := \frac{\Lambda}{m}\bigg((DX + EU + f1_m^T)_+ - X \bigg) \in \mathbb{R}^{h \times m}`, we can show that a Lipschitz constant for the
gradient is

.. math::
    L_{\Theta_2}(X) := \frac{\lambda_{\text{max}}}{m} \max(m,\Vert X \Vert_2^2,\Vert U \Vert_2^2,\Vert X \Vert_2 \Vert U \Vert_2),

where :math:`\lambda_{\text{max}} = \text{max}_{j \in \{1,\cdots,h\}} \lambda_j`. We can then similarly define an 'optimal' step-size :math:`\alpha \Theta_2`.
We have that

.. math::
    \nabla_X \mathcal{G}(\Theta,X) = \frac{1}{m} \bigg\{ A^T(AX + BU + c1_m^T) + (\Lambda - \Lambda D - D^T \Lambda)X + D^T \Lambda (DX + EU + f1_m^T)_+ - \Lambda(EU+f1_m^T) \bigg\}.

A Lipschitz constant for this gradient is

.. math::
    L_X(\Theta) = \frac{1}{m}(\Vert A^TA + \Lambda - \Lambda D + D^T\Lambda \Vert_2 + \lambda_{\text{max}} \Vert D \Vert_2^2).

We can then take the step-size :math:`\alpha_X(\Theta) = \frac{1}{L_X(\Theta)}`. We propose the following block coordinate projected
gradient scheme (BC-gradient) to nd a candidate solution to :eq:`eq_7. We denote compactly the convex set

.. math::
    \mathcal{S}_{\Theta} := \{\Theta \vert \Lambda + A^TA - (\Lambda D + D^T \Lambda) \in \mathbb{s}_+^h, \Vert D \Vert_2 \leq 1 - \epsilon \}

and :math:`\mathcal{P}_{\mathcal{S}_{\Theta}}` the corresponding convex projection

**for** :math:`k = 1, 2, \cdots` **do**

.. math::
    \begin{align}
        \Theta^k &= \mathcal{P}_{\mathcal{S}_{\Theta}}\bigg(\Theta^k - \alpha_{\Theta}(X^{k-1}) \nabla_{\Theta} \mathcal{G}(\Theta^{k-1},X^{k-1}) \bigg) \\
        X^k &= \bigg(X^{k-1} - \alpha_X(\Theta^k) \nabla_X \mathcal{G}(\Theta^k,X^{k-1}) \bigg)
    \end{align}
**end**

3.5 Dual methods
*************************
We propose the following schemes to find an appropriate dual variable :math:`\lambda`. Let :math:`\epsilon > 0` be a precision parameter
for the implicit constraint, i.e. such that we would have

.. math::
    \mathcal{F}(X,DX + EU + f1_m^T) \leq \epsilon

We start with :math:`\lambda = 0` and we solve the two following separate problems

.. math::
    \min_{\color{blue}{X} > 0, \color{blue}{A}, \color{blue}{B}, \color{blue}{c}} \frac{1}{m} \Vert \color{blue}{AX} + \color{blue}{B}U + \color{blue}{c}1_m^T - Y \Vert_F^2

and then

.. math::
    \min_{\color{blue}{D}, \color{blue}{E}, \color{blue}{f}} 1_h^T\mathcal{F}(\color{blue}{X},\color{blue}{DX} + \color{blue}{E}U + \color{blue}{f}1_m^T).

If :math:`\mathcal{F}^* := \mathcal{F}(X,DX + EU + f1_m^T) < \epsilon I_h` then we stop there. Otherwise, we do one of the two following 'dual
updates'

3.5.1 Dual ascent conditional on Fenchel Divergence
*************************
.. math::
    \lambda \leftarrow \lambda + \alpha \mathcal{F}^* \odot 1\{\mathcal{F}^* \geq \epsilon I_h\},
    :label: eq_8

where :math:`\alpha > 0` is a step-size. Note that here we only update the components of :math:`\lambda` for which the corresponding
Fenchel divergence is more than :math:`\epsilon`. We then proceed to solve :eq:`eq_7` using previously discussed methods and
iterate. Alternatively, if the BC-gradient method is used, we can do a dual update after each BC-gradient
update.

3.5.2 Dual variable update conditional on loss
*************************
We start with :math:`\lambda = \epsilon I_h`. Given :math:`(\Theta,X)`, we define the unique :math:`\bar{X}` such that the implicit constraint
is enforced given :math:`\Theta`

.. math::
    \bar{X} = (DX + EU + f1_m^T)_+.

We then define :math:`\Delta X := X - \bar{X}`. We can compute in close form the error on the loss due to the implicit
constraint violation

.. math::
    \begin{align}
        \Delta \mathcal{L} :&= \mathcal{L}(Y,[\Theta,\bar{X}]) - \mathcal{L}(Y,[\Theta,X]) \\
        &= \frac{1}{2m} \bigg(\Vert A \Delta X \Vert_F^2 + Tr(\Omega,A \Delta X) \bigg)
    \end{align}

with :math:`\Omega := BU + c1_m^T`. We can write this error as a sum of contributions with respect to each hidden variable
components :math:`j \in \{1,\cdots,h\}`

.. math::
    \Delta \mathcal{L} = \sum_{j=1}^h \bigg\{ \Delta \mathcal{L}_j := \frac{1}{m} A_j^T \bigg( \frac{1}{2} A \Delta X + \Omega \bigg) \Delta X_j^T \bigg\},

where :math:`A_j \in \mathbb{R}^h` is the :math:`j^{th}` column of :math:`A` and :math:`\Delta X_j \in \mathbb{R}^{1 \times m}` is the :math:`j^{th}`
row of :math:`\Delta X`. The objective of this dual update is to achieve an error on the loss that is smaller than a fraction :math:`\eta \in (0,1)` of the loss

.. math::
    \frac{\Delta \mathcal{L}}{\mathcal{L}(Y,[\Theta,\bar{X}])} \leq \eta.

In order to update each component of the dual variable, we propose the following update. Given :math:`j \in \{1,\cdots,h\}` if

.. math::
    \frac{(\Delta \mathcal{L}_j)_+}{\mathcal{L}(Y,[\Theta,\bar{X}])} \geq \frac{\eta}{h},

then we do the update

.. math::
    \lambda_j \rightarrow \beta \lambda_j,

with :math:`\beta > 1` a hyperparameter.

Loss Functions
*************************
** section 3.2 equation 4 ** + Classification loss
(see more at :ref:`Learning`)

Description of the learning process
*************************************
(see more at :ref:`Formulation`)

Description of the prediction process
**************************************
(see more at :ref:`Prediction`)

Setup
******
TODO

The package is compatible with Python version 3 or higher only.
The user is expected to have installed cvxpy before running the package.
Go to ... for more information.

1. Switch to a proper directory and then type:

::

    git clone + https://github.com/...
