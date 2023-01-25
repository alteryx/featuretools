import numpy as np
import pandas as pd

from featuretools.primitives import Variance


class TestVariance:
    def test_regular(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(variance(np.array([0, 3, 4, 3])), 2.25)

    def test_single(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(variance(np.array([4])), 0)

    def test_double(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(variance(np.array([3, 4])), 0.25)

    def test_empty(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(variance(np.array([])), np.nan)

    def test_nan(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(
            variance(pd.Series([0, np.nan, 4, 3])),
            2.8888888888888893,
        )

    def test_allnan(self):
        variance = Variance().get_function()
        np.testing.assert_almost_equal(
            variance(pd.Series([np.nan, np.nan, np.nan])),
            np.nan,
        )
