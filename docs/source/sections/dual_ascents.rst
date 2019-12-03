.. _Dual Ascents:

Dual Ascents
==================

Dual methods
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

Dual ascent conditional on Fenchel Divergence
*************************
.. math::
    \lambda \leftarrow \lambda + \alpha \mathcal{F}^* \odot 1\{\mathcal{F}^* \geq \epsilon I_h\},
    :label: eq_8

where :math:`\alpha > 0` is a step-size. Note that here we only update the components of :math:`\lambda` for which the corresponding
Fenchel divergence is more than :math:`\epsilon`. We then proceed to solve :eq:`eq_7` using previously discussed methods and
iterate. Alternatively, if the BC-gradient method is used, we can do a dual update after each BC-gradient
update.

Dual variable update conditional on loss
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


Code for calculating Dual Ascents
**********

.. automodule:: utilities.DualAscents
    :members:
