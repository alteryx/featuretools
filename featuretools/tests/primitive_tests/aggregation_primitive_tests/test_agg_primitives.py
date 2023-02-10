from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype

from featuretools.primitives import (
    AverageCountPerUnique,
    FirstLastTimeDelta,
    NMostCommon,
    PercentTrue,
    Trend,
    Variance,
    get_aggregation_primitives,
)


def test_nmostcommon_categorical():
    n_most = NMostCommon(3)
    expected = pd.Series([1.0, 2.0, np.nan])

    ints = pd.Series([1, 2, 1, 1]).astype("int64")
    assert pd.Series(n_most(ints)).equals(expected)

    cats = pd.Series([1, 2, 1, 1]).astype("category")
    assert pd.Series(n_most(cats)).equals(expected)

    # Value counts includes data for categories that are not present in data.
    # Make sure these counts are not included in most common outputs
    extra_dtype = CategoricalDtype(categories=[1, 2, 3])
    cats_extra = pd.Series([1, 2, 1, 1]).astype(extra_dtype)
    assert pd.Series(n_most(cats_extra)).equals(expected)


def test_agg_primitives_can_init_without_params():
    agg_primitives = get_aggregation_primitives().values()
    for agg_primitive in agg_primitives:
        agg_primitive()


def test_trend_works_with_different_input_dtypes():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    numeric = pd.Series([1, 2, 3])

    trend = Trend()
    dtypes = ["float64", "int64", "Int64"]

    for dtype in dtypes:
        actual = trend(numeric.astype(dtype), dates)
        assert np.isclose(actual, 1)


def test_percent_true_boolean():
    booleans = pd.Series([True, False, True, pd.NA], dtype="boolean")
    pct_true = PercentTrue()
    pct_true(booleans) == 0.5


class TestAverageCountPerUnique:
    array = pd.Series([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])

    def test_percent_unique(self):
        primitive_func = AverageCountPerUnique().get_function()
        assert primitive_func(self.array) == 1.25

    def test_nans(self):
        primitive_func = AverageCountPerUnique().get_function()
        array_nans = pd.concat([self.array.copy(), pd.Series([np.nan])])
        assert primitive_func(array_nans) == 1.25
        primitive_func = AverageCountPerUnique(skipna=False).get_function()
        array_nans = pd.concat([self.array.copy(), pd.Series([np.nan])])
        assert primitive_func(array_nans) == (11 / 9.0)

    def test_empty_string(self):
        primitive_func = AverageCountPerUnique().get_function()
        array_empty_string = pd.concat([self.array.copy(), pd.Series([np.nan, "", ""])])
        assert primitive_func(array_empty_string) == (4 / 3.0)


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


class TestFirstLastTimeDelta:
    times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
    actual_delta = (times.iloc[-1] - times.iloc[0]).total_seconds()

    def test_first_last_time_delta(self):
        primitive_func = FirstLastTimeDelta().get_function()
        assert primitive_func(self.times) == self.actual_delta

    def test_with_nans(self):
        primitive_func = FirstLastTimeDelta().get_function()
        times = pd.concat([self.times, pd.Series([np.nan])])
        assert primitive_func(times) == self.actual_delta
        assert pd.isna(primitive_func(pd.Series([np.nan])))
