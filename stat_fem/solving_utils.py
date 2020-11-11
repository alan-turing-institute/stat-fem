import numpy as np
from firedrake import COMM_SELF
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.linear_solver import LinearSolver
from firedrake.vector import Vector
from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix

def solve_forcing_covariance(G, ls, rhs):
    """
    Solve the forcing covariance part of the stat FEM

    This function performs the basic solve needed to determine the
    prior covariance matrix in the stat-fem method. Two solves of 
    the FEM are required, in addition to a sparse matrix multiplication.
    The arguments provide the sparse Forcing Covariance matrix,
    the Firedrake Linear Solver object representing the FEM, and
    the RHS to be solved for this particular iteration.

    Note that this solve function temporarily turns off the BCs for
    the stiffness matrix. This is because Dirichlet BCs will enforce
    strong boundary conditions on the FEM solves, which is not desired
    here.

    :param G: Forcing covariance matrix to be used in the solve.
    :type G: ForcingCovariance
    :param ls: Firedrake Linear Solver to be used in the solve.
    :type ls: Firedrake LinearSolver
    :param rhs: RHS vector to be used in the solve
    :type rhs: Firedrake Vector
    :returns: Solution to :math:`A^{-1}GA^{-1}b` where :math:`A` is the FEM
              stiffness matrix and :math:`b` is the RHS vector.
    :rtype: Firedrake Vector
    """

    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a ForcingCovariance object")
    if not isinstance(ls, LinearSolver):
        raise TypeError("ls must be a firedrake LinearSolver")
    if not isinstance(rhs, Vector):
        raise TypeError("rhs must be a firedrake vector")

    # turn off BC application temporarily

    bcs = ls.A.bcs
    ls.A.bcs = None
    
    rhs_working = rhs.copy()
    x = Function(G.function_space).vector()
    ls.solve(x, rhs_working)
    G.mult(x, rhs_working)
    ls.solve(x, rhs_working)

    # turn BCs back on

    ls.A.bcs = bcs
    
    return x.copy()

def interp_covariance_to_data(im_left, G, ls, im_right, ensemble_comm=COMM_SELF):
    """
    Solve for the interpolated covariance matrix

    Solve for the covariance matrix interpolated to the sensor data locations.
    Note that the arguments allow for two different interpolation matrices
    to be used, in the event that we wish to compute the covariance matrix for
    other locations to make predictions. Note that since the Covariance Matrix
    is symmetric, it is advantageous to put the matrix with fewer spatial
    locations on the right as it will lead to fewer FEM solves (simply
    take the transpose if the reverse order is desired)

    This function solves :math:`\Phi_l^T A^{-1}GA^{-1}\Phi_r`, returning
    it as a 2D numpy array on the root process. This requires doing two FEM
    solves for each location provided in :math:`\Phi_r` plus some sparse
    matrix multiplication. Non-root processes will return an empty 2D
    numpy array (shape ``(0,0)``).

    The solves can be done independently, so optionally a Firedrake
    Ensemble communicator can be provided for dividing up the solves.
    The solves are simply divided up among the processes, so it is up
    to the user to determine an appropriate number of processes given the
    number of sensor locations that are needed. If not provided, the
    solves will be done serially.

    :param im_left: Left side InterpolationMatrix
    :type im_left: InterpolationMatrix
    :param G: Forcing covariance matrix to be used in the solve.
    :type G: ForcingCovariance
    :param ls: Firedrake LinearSolver to be used for the FEM solution
    :type ls: Firedrake LinearSolver
    :param im_right: Right side Interpolation Matrix
    :type im_right: InterpolationMatrix
    :param ensemble_comm: MPI Communicator over which to parallelize the
                          FEM solves (optional, default is solve in series)
    :type ensemble_comm: MPI Communicator
    :returns: Covariance matrix interpolated to sensor locations. If run
              in parallel, this is returned on the root process as a 2D
              numpy array, while all other processes return an empty
              array (shape ``(0,0)``)
    :rtype: ndarray
    """

    if not isinstance(im_left, InterpolationMatrix):
        raise TypeError("first argument to interp_covariance_to_data must be an InterpolationMatrix")
    if not isinstance(ls, LinearSolver):
        raise TypeError("ls must be a firedrake LinearSolver")
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
        tmp = solve_forcing_covariance(G, ls, rhs)
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
