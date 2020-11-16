# stat-fem

Python tools for solving data-constrained finite element problems

[![Build Status](https://travis-ci.com/alan-turing-institute/stat-fem.svg?branch=master)](https://travis-ci.com/alan-turing-institute/stat-fem)
[![codecov](https://codecov.io/gh/alan-turing-institute/stat-fem/branch/master/graph/badge.svg)](https://codecov.io/gh/alan-turing-institute/stat-fem)
[![Documentation Status](https://readthedocs.org/projects/stat-fem/badge/?version=latest)](https://stat-fem.readthedocs.io/en/latest/?badge=latest)

## Overview

This package provides a Python implementation of the Statistical Finite Element Method (FEM) as
described in the paper by Girolami et al. [1] to use data observations to constrain
FEM models. The package builds on top of Firedrake [2] to assemble the underlying FEM system
and uses PETSc [3-4] to perform the sparse linear algebra routines. These tools should allow
the user to create efficient, scalable solvers based on high level Python code to address
challenging problems in data-driven numerical analysis.

## Installation

### Installing Firedrake

`stat-fem` requires a working Firedrake installation. The easiest way to obtain Firedrake is to
follow the installation instructions on the [firedrake homepage](https://www.firedrakeproject.org).

### Installing stat-fem

Once you have installed Firedrake, activate the virtual environment that was created as part of
the installation process. Within the running virtual environment, switch to the main `stat-fem`
directory and proceed with the installation by entering:

```bash
$ pip install -r requirements.txt
$ python setup.py install
```

This will use `pip` to install any missing dependencies (notably Scipy) and install the `stat-fem`
package within the Firedrake virtual environment.

### Using a Docker Container

Alternatively, we provide a working Firedrake Docker container that has the `stat-fem` code
and dependencies installed within the Firedrake virtual environment. See the `docker`
directory in the `stat-fem` repository.

### Testing the installation

The code comes with a full suite of unit tests. Running the test suite uses pytest and pytest-mpi
to collect and run the tests. To run the tests on a single process, simply enter `pytest` into
the running virtual environment from any location in the stat-fem directory. To run the test
suite in parallel, enter `mpiexec -n 2 python -m pytest --with-mpi` or
`mpiexec -n 4 python -m pytest --with-mpi` depending on the number of desired
processes to be used. Tests have only been written for 2 and 4 processes, so
you may get a failure if you attempt to use other choices for the number
of processes.

## Example Scripts

An example illustrating the various code capabilities and features is included in
the `stat-fem/examples` directory.

## Contact

This software was written by Eric Daub as part of a project with the Research Engineering Group at the
Alan Turing Institute.

Any bugs or issues should be filed in the issue tracker on the main Github page.

## References

[1] Mark Girolami, Alastair Gregory, Ge Yin, and Fehmi Cirak. The Statistical Finite Element
    Method. 2019. URL: http://arxiv.org/abs/1905.06391, arXiv:1905.06391.

[2] Florian Rathgeber, David A. Ham, Lawrence Mitchell, Michael Lange, Fabio Luporini,
    Andrew T. T. Mcrae, Gheorghe-Teodor Bercea, Graham R. Markall, and Paul H. J. Kelly.
    Firedrake: automating the finite element method by composing abstractions. *ACM Trans.
    Math. Softw.*, 43(3):24:1–24:27, 2016. URL: http://arxiv.org/abs/1501.01809,
    arXiv:1501.01809, doi:10.1145/2998441.

[3] L. Dalcin, P. Kler, R. Paz, and A. Cosimo, Parallel Distributed Computing using Python,
    Advances in Water Resources, 34(9):1124-1139, 2011.
    http://dx.doi.org/10.1016/j.advwatres.2011.04.013

[4] S. Balay, S. Abhyankar, M. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, A. Dener,
    V. Eijkhout, W. Gropp, D. Karpeyev, D. Kaushik, M. Knepley, D. May, L. Curfman McInnes,
    R. Mills, T. Munson, K. Rupp, P. Sanan, B. Smith, S. Zampini, H. Zhang, and H. Zhang,
    PETSc Users Manual, ANL-95/11 - Revision 3.12, 2019.
    http://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf
