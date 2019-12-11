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

If :math:`\mathcal{F}^* := \mathcal{F}(X,DX + EU + f1_m^T) < \epsilon I_h` then we stop there. Otherwise, we do the following 'dual
update'

Dual ascent conditional on Fenchel Divergence
*************************
.. math::
    \lambda \leftarrow \lambda + \alpha \mathcal{F}^* \odot 1\{\mathcal{F}^* \geq \epsilon I_h\},
    :label: eq_8

where :math:`\alpha > 0` is a step-size. Note that here we only update the components of :math:`\lambda` for which the corresponding
Fenchel divergence is more than :math:`\epsilon`. We then proceed to solve :eq:`eq_7` using previously discussed methods and
iterate. Alternatively, if the BC-gradient method is used, we can do a dual update after each BC-gradient
update.

Code for calculating Dual Ascents
**********

.. automodule:: utilities.DualAscents
    :members:
