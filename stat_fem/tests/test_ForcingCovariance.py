# import numpy as np
# from scipy.spatial.distance import cdist
# from numpy.testing import assert_allclose
# from firedrake import COMM_WORLD
# from firedrake.ensemble import Ensemble
# from firedrake.function import Function
# from firedrake.utility_meshes import UnitIntervalMesh
# from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
# from firedrake.interpolation import interpolate
# from firedrake.petsc import PETSc
# import pytest
# from ..ForcingCovariance import ForcingCovariance
# from .helper_funcs import create_forcing_covariance, create_assembled_problem
# from .helper_funcs import mesh, fs, meshcoords, fc
#
# def test_ForcingCovariance_init(fs, fc):
#     "test init method of ForcingCovariance"
#
#     # note: tests only handle case of a single process
#     # need to think about testing more complicated cases
#
#     # simple example in 1D
#
#     sigma = np.log(1.)
#     l = np.log(0.1)
#
#     n = Function(fs).vector().size()
#     n_local = Function(fs).vector().local_size()
#
#     M = PETSc.Mat().create()
#     M.setSizes(((n_local, -1), (n_local, -1)))
#     M.setFromOptions()
#     M.setUp()
#     start, end = M.getOwnershipRange()
#
#     assert fc.nx == n
#     assert fc.nx_local == n_local
#     assert fc.function_space == fs
#     assert_allclose(fc.sigma, sigma)
#     assert_allclose(fc.l, l)
#     assert_allclose(fc.cutoff, 0.)
#     assert_allclose(fc.regularization, 1.e-8)
#     assert fc.local_startind == start
#     assert fc.local_endind == end
#     assert not fc.is_assembled
#
# def test_ForcingCovariance_init_failures(fs):
#     "situations where ForcingCovariance will fail"
#
#     # bad types for inputs
#
#     sigma = np.log(1.)
#     l = np.log(0.1)
#
#     with pytest.raises(TypeError):
#         ForcingCovariance(1., sigma, l)
#
#     # check that failure occurs with negative regularization
#
#     with pytest.raises(AssertionError):
#         ForcingCovariance(fs, sigma, l, regularization=-1.)
#
# def test_ForcingCovariance_integrate_basis_functions(meshcoords, fc):
#     "test the method to integrate basis functions in ForcingCovariance"
#
#     basis_ordered = np.array([0.05, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.05])
#     basis_expected = np.zeros(11)
#
#     meshcoords_ordered = np.linspace(0., 1., 11)
#
#     for i in range(11):
#         basis_expected[np.where(meshcoords == meshcoords_ordered[i])] = basis_ordered[i]
#
#     assert_allclose(basis_expected, fc._integrate_basis_functions())
#
# def test_ForcingCovariance_compute_G_vals(mesh, fs, fc):
#     "test the compute_G_vals method of Forcing Covariance"
#
#     _, cov_expected = create_forcing_covariance(mesh, fs)
#     nnz_expected = [11]*(fc.local_endind - fc.local_startind)
#
#     G_dict, nnz = fc._compute_G_vals()
#
#     assert nnz == nnz_expected
#
#     for key, val in G_dict.items():
#         assert_allclose(val, cov_expected[key[0], key[1]], atol = 1.e-8, rtol = 1.e-6)
#
#     assert len(G_dict) == len(cov_expected[fc.local_startind:fc.local_endind,:].flatten())
#
# def test_ForcingCovariance_assemble(mesh, fs, fc):
#     "test the generate_G method of Forcing Covariance"
#
#     fc.assemble()
#
#     assert fc.is_assembled
#
#     _, cov_expected = create_forcing_covariance(mesh, fs)
#
#     for i in range(fc.local_startind, fc.local_endind):
#         for j in range(0, 11):
#             assert_allclose(fc.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)
#
#     fc.destroy()
#
# def test_ForcingCovariance_mult(mesh, fs, fc):
#     "test the multiplication method of ForcingCovariance"
#
#     _, cov_expected = create_forcing_covariance(mesh, fs)
#
#     fc.assemble()
#
#     x = Function(fs).vector()
#     x.set_local(np.ones(x.local_size()))
#
#     y = Function(fs).vector()
#
#     fc.mult(x, y)
#
#     ygathered = y.gather()
#
#     assert_allclose(ygathered, np.dot(cov_expected, np.ones(11)))
#
# @pytest.mark.mpi
# @pytest.mark.parametrize("n_proc", [1, 2])
# def test_ForcingCovariance_mult_parallel(n_proc):
#     "test that the multiplication method of ForcingCovariance can be called independently in an ensemble"
#
#     nx = 10
#
#     my_ensemble = Ensemble(COMM_WORLD, n_proc)
#
#     A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)
#     sigma = np.log(1.)
#     l = np.log(0.1)
#     cutoff = 0.
#     regularization = 1.e-8
#
#     fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
#
#     _, cov_expected = create_forcing_covariance(mesh, V)
#
#     fc.assemble()
#
#     if my_ensemble.ensemble_comm.rank == 0:
#
#         x = Function(V).vector()
#         x.set_local(np.ones(x.local_size()))
#
#         y = Function(V).vector()
#
#         fc.mult(x, y)
#
#         ygathered = y.gather()
#
#         assert_allclose(ygathered, np.dot(cov_expected, np.ones(nx + 1)))
#
#     elif my_ensemble.ensemble_comm.rank == 1:
#
#         x = Function(V).vector()
#         x.set_local(0.5*np.ones(x.local_size()))
#
#         y = Function(V).vector()
#
#         fc.mult(x, y)
#
#         ygathered = y.gather()
#
#         assert_allclose(ygathered, np.dot(cov_expected, 0.5*np.ones(nx + 1)))
#
# def test_ForcingCovariance_get_nx(fc):
#     "test the get_nx method of ForcingCovariance"
#
#     assert fc.get_nx() == 11
#
# def test_ForcingCovariance_get_nx_local(fs, fc):
#     "test the get_nx_local method of ForcingCovariance"
#
#     n_local = Function(fs).vector().local_size()
#
#     assert fc.get_nx_local() == n_local
#
# def test_ForcingCovariance_str(fc):
#     "test the string method of ForcingCovariance"
#
#     assert str(fc) == "Forcing Covariance with {} mesh points".format(11)