import numpy as np
from firedrake import COMM_WORLD
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.ensemble import Ensemble
from firedrake.function import Function
from firedrake.functionspace import VectorFunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.interpolation import interpolate
from firedrake.ufl_expr import TestFunction
from firedrake.vector import Vector
from ufl import dx
from firedrake.petsc import PETSc
from .CovarianceFunctions import sqexp

class ForcingCovariance(object):
    "class representing a sparse forcing covariance matrix"
    def __init__(self, function_space, sigma, l, cutoff=1.e-3, regularization=1.e-8,
                 cov=sqexp):
        "create new forcing covariance from a mesh, vector space and covariance function"

        # need to investigate parallelization here, load balancing likely to be uneven
        # if we just use the local ownership from the distributed matrix
        # since each row has an uneven amount of work
        # know that we have reduced bandwidth (though unclear if this translates to a low
        # bandwidth of the assembled covariance matrix)

        if not isinstance(function_space, WithGeometry):
            raise TypeError("bad input type for function_space: must be a FunctionSpace")

        self.function_space = function_space

        self.comm = function_space.comm

        # extract mesh and process local information

        self.nx = Function(self.function_space).vector().size()
        self.nx_local = Function(self.function_space).vector().local_size()

        # set parameters and covariance

        assert regularization >= 0., "regularization parameter must be non-negative"

        self.sigma = sigma
        self.l = l
        self.cutoff = cutoff
        self.regularization = regularization
        self.cov = cov

        # get local ownership information of distributed matrix

        vtemp = PETSc.Vec().create(comm=self.comm)
        vtemp.setSizes((self.nx_local, -1))
        vtemp.setFromOptions()
        vtemp.setUp()

        self.local_startind, self.local_endind = vtemp.getOwnershipRange()

        vtemp.destroy()

        self.is_assembled = False

    def _integrate_basis_functions(self):
        "integrate the basis functions for computing the forcing covariance"

        v = TestFunction(self.function_space)

        return np.array(assemble(Constant(1.) * v * dx).vector().gather())

    def _compute_G_vals(self):
        "compute nonzero values and stores in a dictionary along with number of nonzero elements"

        G_dict = {}
        current_nnz = 0
        nnz = []

        int_basis = self._integrate_basis_functions()
        mesh = self.function_space.ufl_domain()
        W = VectorFunctionSpace(mesh, self.function_space.ufl_element())
        X = interpolate(mesh.coordinates, W)
        meshvals = X.vector().gather()

        for i in range(self.local_startind, self.local_endind):
            diag = (int_basis[i]*int_basis[i]*
                    self.cov(meshvals[i], meshvals[i], self.sigma, self.l)[0,0])
            G_dict[(i, i)] = diag+self.regularization
            current_nnz += 1
            for j in range(0, self.nx):
                if j == i:
                    continue
                new_element = (int_basis[i]*int_basis[j]*
                               self.cov(meshvals[i], meshvals[j],
                                        self.sigma, self.l)[0,0])
                if new_element/diag > self.cutoff:
                    G_dict[(i, j)] = new_element
                    current_nnz += 1
            nnz.append(current_nnz)
            current_nnz = 0

        return G_dict, nnz

    def assemble(self):
        "compute values of G and create sparse matrix"

        if self.is_assembled:
            return

        G_dict, nnz = self._compute_G_vals()

        self.G = PETSc.Mat().create(comm=self.comm)
        self.G.setType('aij')
        self.G.setSizes(((self.nx_local, -1), (self.nx_local, -1)))
        self.G.setPreallocationNNZ(max(nnz))
        self.G.setFromOptions()
        self.G.setUp()

        for key, val in G_dict.items():
            self.G.setValue(key[0], key[1], val)

        self.G.assemble()

        self.is_assembled = True

    def destroy(self):
        "destroy allocated covariance forcing matrix"

        if not self.is_assembled:
            return

        self.G.destroy()

    def mult(self, x, y):
        "perform matrix multiplication with firedrake vectors"

        assert isinstance(x, Vector), "x must be a firedrake vector"
        assert isinstance(y, Vector), "y must be a firedrake vector"

        if not self.is_assembled:
            self.assemble()

        with x.dat.vec_ro as xtmp:
            with y.dat.vec as ytmp:
                self.G.mult(xtmp, ytmp)

    def get_nx(self):
        "return number of nodes for FEM"

        return self.nx

    def get_nx_local(self):
        "get process local number of nodes"

        return self.nx_local

    def __str__(self):
        "create string representation for printing"

        return "Forcing Covariance with {} mesh points".format(self.get_nx())