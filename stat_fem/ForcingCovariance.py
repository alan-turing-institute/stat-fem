import numpy as np
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.functionspaceimpl import WithGeometry
from firedrake.ufl_expr import TestFunction
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

        # assemble integrated basis functions from functionspace
        # need basis functions on all processes, I think -- do we need to do an allgather?

        if not isinstance(function_space, WithGeometry):
            raise TypeError("bad input type for function_space: must be a FunctionSpace")

        self.function_space = function_space

        # extract spatial coordinates from mesh
        # need this information on all processes, how is this stored? need an allgather?

        self.nx = self.function_space.mesh().num_vertices()

        # set parameters and covariance

        assert regularization >= 0., "regularization parameter must be non-negative"

        self.sigma = sigma
        self.l = l
        self.cutoff = cutoff
        self.regularization = regularization
        self.cov = cov

        # get local ownership information of distributed matrix

        G = PETSc.Mat().createSBAIJ((self.nx, self.nx), 1)
        G.setUp()

        self.local_startind, self.local_endind = G.getOwnershipRange()

    def _integrate_basis_functions(self):
        "integrate the basis functions for computing the forcing covariance"

        v = TestFunction(self.function_space)

        return np.array(assemble(Constant(1.) * v * dx).vector().dat.data)

    def _compute_G_vals(self):
        "compute nonzero values and stores in a dictionary along with number of nonzero elements"

        G_dict = {}
        current_nnz = 0
        nnz = []

        int_basis = self._integrate_basis_functions()
        meshvals = np.array(self.function_space.mesh().coordinates.vector().dat.data)

        for i in range(self.local_startind, self.local_endind):
            diag = (int_basis[i]*int_basis[i]*
                    self.cov(meshvals[i], meshvals[i], self.sigma,self.l))
            G_dict[(i, i)] = diag*(1.+self.regularization)
            current_nnz += 1
            for j in range(i+1, self.nx):
                new_element = (int_basis[i]*int_basis[j]*
                               self.cov(meshvals[i], meshvals[j],
                                        self.sigma, self.l))
                if new_element/diag > self.cutoff:
                    G_dict[(i, j)] = new_element
                    current_nnz += 1
            nnz.append(current_nnz)
            current_nnz = 0

        return G_dict, nnz

    def _generate_G(self):
        "compute values of G and create sparse matrix"

        G_dict, nnz = self._compute_G_vals()

        self.G = PETSc.Mat()
        self.G.createSBAIJ((self.nx, self.nx), 1, nnz=nnz)
        self.G.setUp()

        for key, val in G_dict.items():
            self.G.setValue(key[0], key[1], val)

    def assemble(self):
        "compute values of covariance forcing and assemble the sparse matrix"

        self._generate_G()
        self.G.assemble()
