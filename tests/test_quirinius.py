from pytest import mark, raises

import quirinius as qu
import numpy as np

import warnings

pctl50 = np.array([.5])

class Test_ValAtQtl:

    def test_bounds(self):
        nvals = 11
        vals = np.linspace(0., 1., nvals)
        cumul_qtl = (np.linspace(1., nvals, nvals) - 0.5) / nvals
        qtl = np.array([0., 1.])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='numpy.ufunc size changed',
                category=RuntimeWarning)
            vaqs = qu.val_at_qtl_(vals, cumul_qtl, qtl)
        assert np.isnan(vaqs).sum() == 2

    def test_exact(self):
        nvals = 11
        vals = np.linspace(0., 1., nvals)
        cumul_qtl = (np.linspace(1., nvals, nvals) - 0.5) / nvals
        qtl = np.array([0.5])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='numpy.ufunc size changed',
                category=RuntimeWarning)
            q = qu.val_at_qtl_(vals, cumul_qtl, qtl).squeeze()
            assert np.isclose(q, np.median(q))

class Test_wq_:
    pass

class Test_wq:
    pass
