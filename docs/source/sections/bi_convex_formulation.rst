.. _Formulation:

Formulation for IDL
====================

See :ref:`Citing` for in-depth explanation

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

Bi-convex Formulation
***********************************



.. automodule:: utilities.GradientDescents
    :members:
