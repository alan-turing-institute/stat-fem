.. stat-fem documentation master file, created by
   sphinx-quickstart on Mon Mar 23 18:03:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to stat-fem's documentation!
====================================

``stat-fem`` is a Python library for solving data-constrained finite
element analysis problems. It builds on the
`Firedrake <https://www.firedrakeproject.org>`_
Library to handle FEM description and assembly, and the
`PETSc <https://www.mcs.anl.gov/petsc/>`_ library for handling sparse
linear algebra and solving across multiple computer cores. ``stat-fem``
is released under the LGPL open-source license and is free software.
You are free to use, modify, and distribute this software.

This software was written by members of the Alan Turing Institute
Research Engineering Group as part of a project in the Lloyds
Register Foundation Data-Centric Engineering Programme.

The following pages include a brief overview of the software,
instructions for installation, and a walkthrough example illustrating
how it can be used to solve an example problem. The documentation also
outlines how computations can be parallelized to take advantage of
multi-node and multi-core systems.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   intro/overview
   intro/installation
   intro/tutorial
   intro/parallelism
   intro/firedrake21_talk

Details about the ``stat-fem`` implementation can be found on the
following pages:

.. toctree::
   :maxdepth: 2
   :caption: Classes:

   implementation/ForcingCovariance
   implementation/ObsData
   implementation/LinearSolver
   implementation/InterpolationMatrix

.. toctree::
   :maxdepth: 1
   :caption: Additional Functions

   implementation/assemble
   implementation/estimation
   implementation/solving


.. toctree::
   :maxdepth: 1
   :caption: Utilities

   implementation/covariance_functions
   implementation/solving_utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
