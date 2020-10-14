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

## Parallelism

The various solves required for the Statistical FEM can be done in parallel in several ways.

### Parallel base solves

To parallelize the solution of the base FEM model, you can simply run your Python script under MPI
to divide the FEM mesh among the given number of processes. This is handled natively in Firedrake
as well as stat-fem. For instance, depending on your system setup you would enter:

```bash
$ mpiexec -n 4 python model.py
```

The nodes owned by each process are determined when forming the mesh, and this manages all interprocess
communication when performing the matrix inversion to solve the resulting linear system.

While the mesh degrees of freedom in the FEM are divided among the processors, the data is always
held at the root process, as the data dimension is usually significantly less than the number
of degrees of freedom in the mesh and the solves involving the data are unlikely to benefit from
running in parallel due to the communication overhead involved. Therefore, all matrix computations
done in the "data space" that are needed for the solves are done on the root process The collection
of data at root and dispersal of data to the mesh is handled through the interpolation matrix, which
is automatically constructed when performing a solve and should not require any modification when
solving a given problem.

### Parallel Covariance Solves

To compute the FEM model posterior conditioned on the data, one must perform 2 FEM solves per data
constraint. This is the principal computational cost in computing the model posterior once the
other matrices in the system have been assembled. Each data constraint can be handled independently,
and thus this step in the computation can be parallelized.

This parallelization is handled through the Firedrake `Ensemble` class. An ensemble creates
two different MPI communicators: one that parallelizes the base solves (the base communicator,
which is an `Ensemble` class attribute `comm`), and one that parallelizes the covariance solves
(the ensemble communicator, an `Ensemble` class attributed `ensemble_comm`). Each process has a
unique pair of ranks across the two communicators. When creating a Firedrake Ensemble, you must
do so before creating the mesh, and use the base communicator in the ensemble to create the mesh.
The ensemble communicator, if used, is passed to the solving routine when it is called.

When running a script using this type of parallelism, the user must specify how to divide the
MPI processes across the two different types of solves. This is done when constructing the
`Ensemble` instance. For example, if you wish you use 4 processes total, with 2 dedicated to each
base solve and 2 processes over which the data constraint solves will be done, you would do the
following in the file `model.py`:

```python
from firedrake import Ensemble, COMM_WORLD, UnitSquareMesh

my_ensemble = Ensemble(COMM_WORLD, 2)

# divide mesh across two processes in base comm
mesh = UnitSquareMesh(51, 51, comm=my_ensemble.comm)

...
```

Then run the python script with `mpiexec -n 4 python model.py`

Note that it is up to the user to ensure that the total number of processes is divisible by the base
number of processes. One way to handle this more robustly would be to use input arguments to manage
the number of processes in a particular script.

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
