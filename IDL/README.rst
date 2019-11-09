.. -*- mode: rst -*-


IDL - a scikit-learn package for Implicit Deep Learning
============================================================

Hey guys, hint to organize our work in the package :

Structure of the code
----------------------
* The core code is in IDLtemplate, and the main class in IDL.py
* IDLtemplate/tests contains standardized tests to verify that our package can be a "sklearn" one
* doc contains documentations related to implementing the whole sklearn package
* example will contain some straightforward example At the End.

Rules of coding
----------------
* Only the code in IDLtemplate/utilities should be overwritten
* To the best of possible, do not test your functions inside the core functions. Instead use the folder IDLtemplate/utilities/draft_examples


Documentation
----------------

.. _documentation: https://sklearn-template.readthedocs.io/en/latest/quick_start.html

Refer to the documentation_ to modify the template for your own scikit-learn
contribution.

