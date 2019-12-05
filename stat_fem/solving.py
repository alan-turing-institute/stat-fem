import numpy as np
from firedrake import COMM_WORLD
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.matrix import Matrix
from firedrake.vector import Vector
from firedrake.solving import solve
from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix
from .ObsData import ObsData
from .ModelDiscrepancy import ModelDiscrepancy
from .solving_utils import _solve_forcing_covariance

def solve_conditioned_FEM(A, x, b, G, M, data, ensemble=COMM_WORLD):
    "Solve for the FEM posterior conditioned on the data"

    if not isinstance(A, Matrix):
       raise TypeError("A must be a firedrake matrix")
    if not isinstance(x, (Function, Vector)):
        raise TypeError("x must be a firedrake function or vector")
    if not isinstance(b, (Function, Vector)):
        raise TypeError("b must be a firedrake function or vector")
    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a forcing covariance")
    if not isinstance(M, ModelDiscrepancy):
        raise TypeError("M must be a model discrepancy")
    if not isinstance(data, ObsData):
        raise TypeError("data must be an ObsData type")

    # solve base FEM

    solve(A, x, b)

    # solve for posterior FEM solution

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords(), comm=ensemble)

    # invert model discrepancy and interpolate into mesh space

    tmp_dataspace_1 = np.linalg.solve(M.get_K_plus_sigma(), data.get_data())
    tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_1)

    # solve forcing covariance and interpolate to dataspace

    tmp_meshspace_2 = _solve_forcing_covariance(G, A, tmp_meshspace_1)+x.vector()
    tmp_dataspace_1 = im.interp_mesh_to_data(tmp_meshspace_2)

    # solve model discrepancy plus forcing covariance system and interpolate into meshspace

    tmp_dataspace_2 = np.linalg.solve(M.get_K_plus_sigma() + im.interp_covariance_to_data(G, A), tmp_dataspace_1)
    tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_2)

    # deallocate interpolation matrix
    im.destroy()

    # solve final covariance system and return result

    return (tmp_meshspace_2 - _solve_forcing_covariance(G, A, tmp_meshspace_1)).function

def solve_conditioned_FEM_dataspace(A, x, b, G, M, data, ensemble=COMM_WORLD):
    "solve for conditioned fem plus covariance in the data space"

    # solve base FEM

    solve(A, x, b)

    # solve for posterior FEM solution

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords(), comm=ensemble)

    # form interpolated prior covariance and solve for posterior covariance

    Cu = im.interp_covariance_to_data(G, A)

    Kinv = np.linalg.inv(M.get_K_plus_sigma())
    Cinv = np.linalg.inv(Cu)
    Cuy = np.linalg.inv(Cinv + Kinv)

    tmp = im.interp_mesh_to_data(x.vector())

    # get posterior mean

    muy = np.dot(Cuy, np.dot(Kinv, data.get_data()) + np.dot(Cinv, tmp))

    im.destroy()

    return muy, Cuy
