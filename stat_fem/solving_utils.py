import numpy as np
from firedrake import COMM_WORLD
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.matrix import Matrix
from firedrake.vector import Vector
import firedrake.solving
from .ForcingCovariance import ForcingCovariance

def _solve_forcing_covariance(G, A, rhs):
    "solve the forcing covariance part of the stat FEM"

    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a ForcingCovariance object")
    if not isinstance(A, Matrix):
        raise TypeError("A must be an assembled firedrake matrix")
    if not isinstance(rhs, Vector):
        raise TypeError("rhs must be a firedrake vector")

    ksp = PETSc.KSP().create()
    ksp.setOperators(A.petscmat)
    ksp.setFromOptions()

    # call the necessary solves and multiplications
    # alternately overwrite rhs_working and x

    rhs_working = rhs.copy()
    x = Function(G.function_space).vector()
    with rhs_working.dat.vec_ro as rhs_temp:
        with x.dat.vec as x_temp:
            ksp.solve(rhs_temp, x_temp)
    G.mult(x, rhs_working)
    with rhs_working.dat.vec_ro as rhs_temp:
        with x.dat.vec as x_temp:
            ksp.solve(rhs_temp, x_temp)

    # final solution is in x, return that

    return x.copy()
