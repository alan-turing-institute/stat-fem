from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix
import firedrake.assemble

def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             mat_type=None, sub_mat_type=None,
             appctx={}, options_prefix=None, **kwargs):
    """
    Overloaded assembly function to include assembly of stat-fem operators

    This provides a functional interface to assemblying stat-fem objects in the
    same manner as other matrices and vectors in Firedrake. If the input object
    is a ``ForcingCovariance`` or ``InterpolationMatrix`` object, it calls the
    ``assemble`` method and returns. Otherwise, it passes the arguments along
    to Firedrake for assembly.

    If f is a :class:`~ufl.classes.Form` then this evaluates the corresponding
    integral(s) and returns a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` or :class:`.ImplicitMatrix`
    for 2-forms.

    If f is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If f is a Slate tensor expression, then it will be compiled using Slate's
    linear algebra compiler.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``f`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``f`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    :param f: a :class:`stat_fem.ForcingCovariance`, :class:`~stat_fem.InterpolationMatrix`,
       :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or :class:`~slate.TensorBase`
       expression.
    :param tensor: an existing tensor object to place the result in
      (optional).
    :param bcs: a list of boundary conditions to apply (optional).
    :param form_compiler_parameters: (optional) dict of parameters to pass to
      the form compiler.  Ignored if not assembling a
      :class:`~ufl.classes.Form`.  Any parameters provided here will be
      overridden by parameters set on the :class:`~ufl.classes.Measure` in the
      form.  For example, if a ``quadrature_degree`` of 4 is
      specified in this argument, but a degree of 3 is requested in
      the measure, the latter will be used.
    :param mat_type: (optional) string indicating how a 2-form (matrix) should be
      assembled -- either as a monolithic matrix ('aij' or 'baij'), a block matrix
      ('nest'), or left as a :class:`.ImplicitMatrix` giving matrix-free
      actions ('matfree').  If not supplied, the default value in
      ``parameters["default_matrix_type"]`` is used.  BAIJ differs
      from AIJ in that only the block sparsity rather than the dof
      sparsity is constructed.  This can result in some memory
      savings, but does not work with all PETSc preconditioners.
      BAIJ matrices only make sense for non-mixed matrices.
    :param sub_mat_type: (optional) string indicating the matrix type to
      use *inside* a nested block matrix.  Only makes sense if
      ``mat_type`` is ``nest``.  May be one of 'aij' or 'baij'.  If
      not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :param appctx: Additional information to hang on the assembled
      matrix if an implicit matrix is requested (mat_type "matfree").
    :param options_prefix: PETSc options prefix to apply to matrices.
    """

    if isinstance(f, (ForcingCovariance, InterpolationMatrix)):
        f.assemble()
        return f
    else:
        return firedrake.assemble(f, tensor, bcs, form_compiler_parameters,
                                  mat_type, sub_mat_type, appctx, options_prefix, **kwargs)


