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
from .covariance_functions import sqexp

class ForcingCovariance(object):
    """
    Class representing a sparse forcing covariance matrix

    This class represents a forcing covariance matrix, which describes uncertainty in the
    underlying forcing function of the FEM solution to the PDE. In large simulations,
    this matrix cannot be stored as a dense matrix, and thus we approximate this as a
    sparse matrix. The current implementation does this by cutting off terms that are
    less than the diagonal by a certain factor, which is controlled by the ``cutoff``
    attribute. The covariance matrix is usually regularized by adding a small nugget
    to the diagonal to ensure that it is well behaved. The implementation also makes the
    assumption that the correlation function can be assumed to be approximately constant
    over an entire mesh cell. This is effectively assuming that the correlation length is
    much longer than the mesh spacing, which is typically the case of interest in this
    type of problem.

    The class is effectively a wrapper to an underlying sparse PETSc ``'aij'`` Compressed
    Row Storage (CSR) matrix. The main operation that it supports is multiplication with
    a column vector, which must be a Firedrake ``Vector`` class. This operation is used
    in the data-conditioned FEM solves. This is exposed via the ``mult`` method, which
    behaves like the underlying PETSc multiplication method.

    Because the sparsity of the matrix is not known ahead of time and the way that PETSc
    handles memory allocation, this currently limits the size of problems that can reasonably
    be addressed, as it must compute all matrix elements before cutting off the matrix and
    store the nonzero elements in memory prior to allocating the PETSc matrix. This
    is done row-by-row at the moment, which means that a processor must store several
    vectors over the entire global mesh in order to compute the local rows of the matrix.

    Parameters required to initialize a new ``ForcingCovariance`` object are the function
    space over which the FEM solution will be computed, the standard deviation and
    correlation length scale used in computing the covariance function, and optional
    parameters to specify the cutoff value for making the covariance matrix sparse
    and regularization nugget to be added. Note that the default parameters may not work
    well for your particular problem -- in particular, if the cutoff is too small there
    is a good chance you end up with a dense matrix that may require too much memory.

    """
    def __init__(self, function_space, sigma, l, cutoff=1.e-3, regularization=1.e-8,
                 cov=sqexp):
        """
        Create new forcing covariance

        Creates a new ForcingCovariance object from a function space, parameters, and
        covariance function. Required parameters are the function space and sigma and
        correlation length parameters needed to compute the covariance matrix.

        Note that this just initializes the object, and does not compute the matrix
        entries or assemble the final PETSc matrix. This is done using the ``assemble``
        method, though if you attempt to use an unassembled matrix assembly will
        automatically be done. However the domain decomposition is done here to determine
        the number of DOFs handled by each process.
        """

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

        self.G = None
        
    def _integrate_basis_functions(self):
        """
        Integrate the basis functions for computing the forcing covariance

        Assembly requires doing a double inner product with the covariance function and the basis
        functions. Because the covariance function is approximate already, we simplify this assembly
        by assuming that this inner product is just the integrated individual basis functions
        multiplied by the covariance function value at the particular mesh location. Returns
        the values for the entire mesh as a numpy array on all processes (as the values for the
        entire mesh are needed on each process to compute the individual matrix rows).

        :returns: Integrated basis functions for each degree of freedom for all processes
        :rtype: ndarray
        """

        v = TestFunction(self.function_space)

        return np.array(assemble(Constant(1.) * v * dx).vector().gather())

    def _compute_G_vals(self):
        """
        Compute nonzero values of the covariance matrix and the number of nonzero elements

        This method pre-computes the values of the covariance function, storing those above the
        cutoff. The nonzero elements for each row are stored in a dictionary and returned with
        the maximum number found over all rows for this process. This is needed to allocate
        memory for the sparse matrix and fill with the computed rows.

        :returns: tuple holding a dictionary mapping row indices to a tuple containing the
                  matrix entries (as a numpy array) and the integer indices of the columns
                  represented by those matrix entries.
        :rtype: tuple
        """

        G_dict = {}
        nnz = 0

        int_basis = self._integrate_basis_functions()
        mesh = self.function_space.ufl_domain()
        W = VectorFunctionSpace(mesh, self.function_space.ufl_element())
        X = interpolate(mesh.coordinates, W)
        meshvals = X.vector().gather()
        if X.dat.data.ndim == 2:
            meshvals = np.reshape(meshvals, (-1, X.dat.data.shape[1]))
        else:
            meshvals = np.reshape(meshvals, (-1, 1))
        assert meshvals.shape[0] == self.nx, "error in gathering mesh coordinates"

        for i in range(self.local_startind, self.local_endind):
            row = (int_basis[i]*int_basis*
                   self.cov(meshvals[i], meshvals, self.sigma, self.l))[0]
            row[i] += self.regularization
            above_cutoff = (row/row[i] > self.cutoff)
            G_dict[i] = (row[above_cutoff], np.arange(0, self.nx, dtype=PETSc.IntType)[above_cutoff])
            new_nnz = int(np.sum(above_cutoff))
            if new_nnz > nnz:
                nnz = new_nnz

        return G_dict, nnz

    def assemble(self):
        """
        Compute values of G and assemble the sparse matrix

        This method creates the underlying sparse covariance matrix based on the pre-computed entries,
        allocates memory, sets the values, and finalizes assembly. No inputs, no return value. If this
        method is called and the matrix has already been assembled, it will return with no effect.
        """

        if self.is_assembled:
            return

        G_dict, nnz = self._compute_G_vals()

        self.G = PETSc.Mat().create(comm=self.comm)
        self.G.setType('aij')
        self.G.setSizes(((self.nx_local, -1), (self.nx_local, -1)))
        self.G.setPreallocationNNZ(nnz)
        self.G.setFromOptions()
        self.G.setUp()

        for key, val in G_dict.items():
            self.G.setValues(np.array(key, dtype=PETSc.IntType), val[1], val[0])

        self.G.assemble()

        self.is_assembled = True

    def destroy(self):
        """
        Destroy allocated covariance forcing matrix

        Deallocates memory for an assembled forcing matrix. If memory has not been allocated, returns
        with no effect. No inputs, no return value.
        """

        if not self.is_assembled:
            return

        self.G.destroy()

    def mult(self, x, y):
        """
        Perform matrix multiplication with firedrake vector

        The principal operation done with a ``ForcingCovariance`` object is matrix multiplication
        with a vector. This method is a wrapper to the underlying method for the PETSc object, which
        modifies the ``y`` vector argument in-place. The provided vectors must be Firedrake ``Vector``
        objects.

        If the underlying covariance sparse matrix has not been assembled, it will be assembled prior
        to performing the matrix multiplication.

        :param x: Firedrake vector that will be multiplied with the covariance matrix. Must be a distributed
                  vector with the same row allocations as the covariance matrix.
        :type x: Vector
        :param y: Firedrake vector that will hold the result of multiplying the input vector with the
                  covariance matrix. Must be a distributed vector with the same row allocations as the
                  covariance matrix. Modified in place.
        :type y: Vector
        :returns: None
        """

        assert isinstance(x, Vector), "x must be a firedrake vector"
        assert isinstance(y, Vector), "y must be a firedrake vector"

        if not self.is_assembled:
            self.assemble()

        with x.dat.vec_ro as xtmp:
            with y.dat.vec as ytmp:
                self.G.mult(xtmp, ytmp)

    def get_nx(self):
        """
        Return number of global nodal DOFs for FEM

        Returns the number of global nodal DOFs for the FEM over all processes.

        :returns: number of global DOFs
        :rtype: int
        """

        return self.nx

    def get_nx_local(self):
        """
        Return number of local nodal DOFs for FEM

        Returns the number of local nodal DOFs for the FEM for the current process.

        :returns: number of local DOFs
        :rtype: int
        """

        return self.nx_local

    def __str__(self):
        """
        Create a string representation for printing

        :returns: String representation
        :rtype: str
        """

        return "Forcing Covariance with {} mesh points".format(self.get_nx())
