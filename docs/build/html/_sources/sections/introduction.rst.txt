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

** section 1.1 **

Given an input :math:`u \in \mathbb{R}^n`, where n denotes the number of features,
we define the implicit deep learning prediction rule :math:`\hat{y}(u) \in \mathbb{R}^n` with ReLU activation

.. math::
    \hat{y}(u) = ...


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
