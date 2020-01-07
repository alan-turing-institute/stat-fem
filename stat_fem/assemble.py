from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix
import firedrake.assemble

def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             inverse=False, mat_type=None, sub_mat_type=None,
             appctx={}, options_prefix=None, **kwargs):
    "overloaded assembly function to include assembly of stat-fem operators"

    if isinstance(f, (ForcingCovariance, InterpolationMatrix)):
        f.assemble()
        return f
    else:
        return firedrake.assemble(f, tensor, bcs, form_compiler_parameters, inverse,
                                  mat_type, sub_mat_type, appctx, options_prefix, **kwargs)


