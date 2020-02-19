import numba as nb
from numba.extending import overload, register_jitable

import numpy as np

import itertools

from .exceptions import *

@nb.njit
def val_at_qtl(vals, cumul_qtl, qtl):
    """computes interpolated value at given quantile(s)
    
    equivalent to scipy scoreatpercentile
    
    Parameters
    ----------
    vals : {|ndarray|}
        values to be weighted, pre-sorted
    cumul_qtl : {|ndarray|}
        cumulative quantiles of values
    qtl : {|ndarray|}
        quantile(s) sought
    
    Returns
    -------
    np.ndarray
        interpolated values at specified quantiles
    """

    n = vals.size

    qtl = np.atleast_1d(qtl)

    # search in cumulative quantile array for values bounding target quantile
    ix_lhs = np.searchsorted(cumul_qtl, qtl)
    ix_rhs = ix_lhs + 1

    # if bounding indices are out of range, set to counterpart
    outofbounds_l = (ix_lhs <= 0)
    outofbounds_h = (ix_rhs >= n)
    outofbounds = outofbounds_l | outofbounds_h
    ix_rhs[outofbounds_h] -= 1
    ix_lhs[outofbounds_l] += 1

    # get values and quantiles bounding target
    val_lhs, val_rhs = vals[ix_lhs], vals[ix_rhs]
    qtl_lhs, qtl_rhs = cumul_qtl[ix_lhs], cumul_qtl[ix_rhs]

    # interpolated value: position between bound values
    qtl_frac_btwn_nghbrs = (qtl - qtl_lhs) / (qtl_rhs - qtl_lhs)
    dval_btwn_nghbrs = val_rhs - val_lhs
    val_at_qtl = val_lhs + qtl_frac_btwn_nghbrs * dval_btwn_nghbrs

    val_at_qtl[outofbounds] = np.nan

    return val_at_qtl

@nb.njit
def wq(vals, wts, qtls, order=None, mask=None):
    """user-facing version of `wq_`, which broadcasts
    
    broadcasts functionality of wq_ against all axes of wts,
        *except the final one*
    
    Parameters
    ----------
    vals : {|ndarray|}
        array of values to be weighted, the same for all 1d slices
    wts : {|ndarray|}
        array of weights to be broadcasted against `vals`
    qtls : {|ndarray|}
        quantile(s) sought
    order : {|ndarray| of int}, optional
        index order of `vals` (the default is None, which triggers autosorting)
    mask : {|ndarray|, None}, optional
        mask array or None (the default is None, which suppresses masking behavior)
    
    Returns
    -------
    np.ndarray
        array of quantiles, where first (n-1) dims are same as first (n-1) of
        wts, and the final is equal to the length of `qtls`
    """

    # order if not ordered
    if order is None:
        order = np.argsort(vals)

    # if weights are simple 1d, then just pass straight to wq_
    if wts.ndim == 1:
        return wq_(vals, wts, qtls, order)
    else:
        # call _wq over all indices, broadcasting over last axis
        nqtls = qtls.size
        mapped_shape = wts.shape[:-1]
        ret_shape = (*mapped_shape, nqtls)
        ret = np.nan * np.ones(ret_shape)

        for nd_index in np.ndindex(mapped_shape):
            if mask is None:
                pass
            elif mask.__getitem__(nd_index):
                continue

            ret[nd_index] = wq_(vals, wts[nd_index], qtls, order)

        return ret


@nb.njit(parallel=True)
def wq_(vals, wts, qtls, order):
    """weighted quantile base function
    
    LLVM, JIT-compiled function for finding weighted quantile
    
    Parameters
    ----------
    vals : {|ndarray|}
        array of values to be weighted
    wts : {|ndarray|}
        array of weights
    qtls : {|ndarray|}
        number in (0, 1), or 1-D iterable thereof
    order : {|ndarray|}
        array describing ordering of vals
    """

    # put values and weights in ascending order
    vals_ordered = vals[order]
    wts_ordered = wts[order]

    wt_tot = wts.sum()

    # if all weights are low, return nan
    if not wt_tot > 0.:
        nan = np.nan
        quantiles = np.nan * np.ones(len(qtls))
    else:
        cumul_qtl = (np.cumsum(wts_ordered) - 0.5 * wts_ordered) / wt_tot
        
        quantiles = val_at_qtl(vals_ordered, cumul_qtl, qtls)

    return quantiles
