.. _parallelism:

Parallelism in stat-fem
=======================

The various solves required for the Statistical FEM can be done in parallel in several ways.

Parallel base solves
--------------------

To parallelize the solution of the base FEM model, you can simply run your Python script under MPI
to divide the FEM mesh among the given number of processes. This is handled natively in Firedrake
as well as ``stat-fem``. For instance, depending on your system setup you would enter the following
into the shell to execute a script in parallel: ::

  $ mpiexec -n 4 python model.py

The nodes owned by each process are determined when forming the mesh, and this manages all interprocess
communication when performing the matrix inversion to solve the resulting linear system.

While the mesh degrees of freedom in the FEM are divided among the processors, the data is always
held at the root process, as the data dimension is usually significantly less than the number
of degrees of freedom in the mesh and the solves involving the data are unlikely to benefit from
running in parallel due to the communication overhead involved. Therefore, all matrix computations
done in the "data space" that are needed for the solves are done on the root process. The collection
of data at root and dispersal of data to the mesh is handled through the interpolation matrix, which
is automatically constructed when performing a solve and should not require any modification when
solving a given problem.

Parallel Covariance Solves
--------------------------

To compute the FEM model posterior conditioned on the data, one must perform 2 FEM solves per data
constraint. This is the principal computational cost in computing the model posterior once the
other matrices in the system have been assembled. Each data constraint can be handled independently,
and thus this step in the computation can be parallelized.

This parallelization is handled through the Firedrake ``Ensemble`` class. An ensemble creates
two different MPI communicators: one that parallelizes the base solves (the base communicator,
which is an ``Ensemble`` class attribute ``comm``), and one that parallelizes the covariance solves
(the ensemble communicator, an ``Ensemble`` class attribute ``ensemble_comm``). Each process has a
unique pair of ranks across the two communicators. When creating a Firedrake Ensemble, you must
do so before creating the mesh, and use the base communicator in the ensemble to create the mesh.
The ensemble communicator, if used, is passed to the solving routine or the ``LinearSolver``
object when it is created.

When running a script using this type of parallelism, the user must specify how to divide the
MPI processes across the two different types of solves. This is done when constructing the
``Ensemble`` instance. For example, if you wish you use 4 processes total, with 2 dedicated to each
individual solve and 2 processes over which the data constraint solves will be divided, you would do the
following in the file ``model.py``: ::

  from firedrake import Ensemble, COMM_WORLD, UnitSquareMesh, Function

  my_ensemble = Ensemble(COMM_WORLD, 2)

  # divide mesh across two processes in base comm
  mesh = UnitSquareMesh(51, 51, comm=my_ensemble.comm)

  # Code to assemble stiffness matrix, RHS, BCs, forcing covariance, data, etc
  # see stat_fem/examples/poisson.py for an example of how to create and
  # assemble all of these objects 

  ...

  from stat_fem import solve_posterior

  # A is an assembled matrix, x is a function in the desired function space,
  # b is the RHS, fc is the assembled ForcingCovariance object, and
  # data is the ObsData object

  # this will divide the solves over the 2 processors in the ensemble_comm
  
  solve_posterior(A, x, b, fc, data, ensemble_comm=my_ensemble.ensemble_comm) 
  
Then run the python script with ``mpiexec -n 4 python model.py``. This will work similarly
for all solving functions, a ``LinearSolver`` object, or the ``estimate_params_MAP`` function,
all of which accept an ``ensemble_comm`` keyword argument.

Note that it is up to the user to ensure that the total number of processes is divisible by the base
number of processes used when creating the ``Ensemble`` object.

