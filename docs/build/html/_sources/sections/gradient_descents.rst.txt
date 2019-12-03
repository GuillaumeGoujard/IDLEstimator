.. _Gradient Descents:

Gradient Descents
==================

Block coordinate descent and first order methods
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

Bi-Convexity of the Loss function
**********************************
TODO


Gradient Descents
*************************
TODO


Code for calculating Gradient Descent
*************************
.. automodule:: utilities.GradientDescents
    :members:
