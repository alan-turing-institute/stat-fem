import numpy as np
from firedrake import COMM_WORLD, COMM_SELF
from firedrake.ensemble import Ensemble
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.matrix import Matrix
from firedrake.vector import Vector
import firedrake.solving
from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix

def solve_forcing_covariance(G, A, rhs):
    "solve the forcing covariance part of the stat FEM"

    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a ForcingCovariance object")
    if not isinstance(A, Matrix):
        raise TypeError("A must be an assembled firedrake matrix")
    if not isinstance(rhs, Vector):
        raise TypeError("rhs must be a firedrake vector")

    # create Krylov solver and attach stiffness matrix
    # to do: allow user to customize by passing arguments
    # and preconditioner to ksp class

    ksp = PETSc.KSP().create(comm=G.comm)
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

def interp_covariance_to_data(im_left, G, A, im_right, ensemble_comm=COMM_SELF):
    "solve for the interpolated covariance matrix"

    if not isinstance(im_left, InterpolationMatrix):
        raise TypeError("first argument to interp_covariance_to_data must be an InterpolationMatrix")
    if not isinstance(A, Matrix):
        raise TypeError("A must be an assembled firedrake matrix")
    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a ForcingCovariance class")
    if not isinstance(im_right, InterpolationMatrix):
        raise TypeError("fourth argument to interp_covariance_to_data must be an InterpolationMatrix")
    if not isinstance(ensemble_comm, type(COMM_SELF)):
        raise TypeError("ensemble_comm must be an MPI communicator created from a firedrake Ensemble")

    # use ensemble comm to split up solves across ensemble processes

    v_tmp = PETSc.Vec().create(comm=ensemble_comm)
    v_tmp.setSizes((-1, im_right.n_data))
    v_tmp.setFromOptions()

    imin, imax = v_tmp.getOwnershipRange()

    v_tmp.destroy()

    # create array for holding results
    # if root on base comm, will have data at the end of the solve/interpolation
    # otherwise, size will be zero

    if im_left.comm.rank == 0:
        n_local = im_left.n_data
    else:
        n_local = 0

    # additional index is for the column vectors that this process owns in the
    # ensemble, which has length imax - imin

    result_tmparray = np.zeros((imax - imin, n_local))

    for i in range(imin, imax):
        rhs = im_right.get_meshspace_column_vector(i)
        tmp = solve_forcing_covariance(G, A, rhs)
        result_tmparray[i - imin] = im_left.interp_mesh_to_data(tmp)

    # create distributed vector for gathering results at root

    cov_distrib = PETSc.Vec().create(comm=ensemble_comm)
    cov_distrib.setSizes((n_local*(imax - imin), -1))
    cov_distrib.setFromOptions()

    cov_distrib.array = result_tmparray.flatten()

    scatterfunc, cov_gathered = PETSc.Scatter.toZero(cov_distrib)

    scatterfunc.scatter(cov_distrib, cov_gathered,
                        mode=PETSc.ScatterMode.SCATTER_FORWARD)

    out_array = np.copy(cov_gathered.array)
    cov_distrib.destroy()
    cov_gathered.destroy()

    # reshape output -- if I am root on both the main comm and ensemble comm then
    # I have the whole array. Other processes have nothing

    if im_left.comm.rank == 0 and ensemble_comm.rank == 0:
        outsize = (im_left.n_data, im_right.n_data)
    else:
        outsize = (0,0)

    return np.reshape(out_array, outsize)