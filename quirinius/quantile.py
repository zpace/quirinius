import numba as nb
import numpy as np

@nb.njit
def val_at_qtl_(vals, cumul_qtl, qtl):

    n = len(vals)

    # search in cumulative quantile array for values bounding target quantile
    ix_lhs = np.searchsorted(cumul_qtl, qtl)
    ix_rhs = ix_lhs + 1

    # if bounding indices are out of range, set to counterpart
    outofbounds = np.logical_or((ix_lhs <= 0), (ix_rhs >= n))

    # get values and quantiles bounding target
    val_lhs, val_rhs = vals[ix_lhs], vals[ix_rhs]
    qtl_lhs, qtl_rhs = cumul_qtl[ix_lhs], cumul_qtl[ix_rhs]

    # interpolated value: position between bound values
    qtl_frac_btwn_nghbrs = (qtl - qtl_lhs) / (qtl_rhs - qtl_lhs)
    dval_btwn_nghbrs = val_rhs - val_lhs
    val_at_qtl = val_lhs + qtl_frac_btwn_nghbrs * dval_btwn_nghbrs

    val_at_qtl[outofbounds] = np.nan

    return val_at_qtl


@nb.njit(parallel=True)
def wq_(vals, wts, qtls, order):
    """weighted quantile base function
    
    LLVM, JIT-compiled function for finding weighted quantile
    
    Parameters
    ----------
    vals : {np.ndarray}
        array of values to be weighted
    wts : {np.ndarray}
        array of weights
    qtls : {np.ndarray}
        number in (0, 1), or 1-D iterable thereof
    order : {np.ndarray}
        array describing ordering of vals
    """

    # put values and weights in ascending order
    vals_ordered = vals[order]
    wts_ordered = wts[order]

    wt_tot = wts.sum()

    # if all weights are low, return nan
    if not wt_tot > 0.:
        quantiles = np.nan * np.ones_like(qtls)
    else:
        cumul_qtl = (np.cumsum(wts_ordered) - 0.5 * wts_ordered) / wt_tot
        
        quantiles = val_at_qtl_(vals_ordered, cumul_qtl, qtls)

    return quantiles

