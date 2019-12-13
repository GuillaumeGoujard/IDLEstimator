.. -*- mode: rst -*-


IDL - a scikit-learn package for Implicit Deep Learning
============================================================

This package can be used to fit an Implicit Deep Learning (IDL) model for regression
and classification purpose.

The IDL.fit function estimates a vector of parameters by applying successively
gradient descents (see more at :ref:`Gradient Descents`) and dual ascent
(see more at :ref:`Dual Ascents`).

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


Structure of the code
----------------------
* The IDLModel class is in IDLtemplate/IDL.py
* Working examples in are in IDLtemplate/tests


Contributors
--------------

Erik Boee, Marius Landsverk \& Axel Roenold

Citing
######

* "Implicit Deep Learning" Laurent El Ghaoui, Fangda Gu, Bertrand Travacca, Armin Askari, arXiv:1908.06315, 2019.
* "Bi-convex Implicit Deep Learning" Bertrand Travacca, October 2019


