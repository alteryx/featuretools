from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from pytest import raises

from featuretools.primitives import (
    Autocorrelation,
    AverageCountPerUnique,
    Correlation,
    DateFirstEvent,
    FirstLastTimeDelta,
    NMostCommon,
    PercentTrue,
    Trend,
    Variance,
    get_aggregation_primitives,
)
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    test_serialize,
    valid_dfs,
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


class TestAverageCountPerUnique(PrimitiveTestBase):
    primitive = AverageCountPerUnique
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

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive.name.upper())


class TestVariance(PrimitiveTestBase):
    primitive = Variance

    def test_regular(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(variance(np.array([0, 3, 4, 3])), 2.25)

    def test_single(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(variance(np.array([4])), 0)

    def test_double(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(variance(np.array([3, 4])), 0.25)

    def test_empty(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(variance(np.array([])), np.nan)

    def test_nan(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(
            variance(pd.Series([0, np.nan, 4, 3])),
            2.8888888888888893,
        )

    def test_allnan(self):
        variance = self.primitive().get_function()
        np.testing.assert_almost_equal(
            variance(pd.Series([np.nan, np.nan, np.nan])),
            np.nan,
        )


class TestFirstLastTimeDelta(PrimitiveTestBase):
    primitive = FirstLastTimeDelta
    times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
    actual_delta = (times.iloc[-1] - times.iloc[0]).total_seconds()

    def test_first_last_time_delta(self):
        primitive_func = self.primitive().get_function()
        assert primitive_func(self.times) == self.actual_delta

    def test_with_nans(self):
        primitive_func = self.primitive().get_function()
        times = pd.concat([self.times, pd.Series([np.nan])])
        assert primitive_func(times) == self.actual_delta
        assert pd.isna(primitive_func(pd.Series([np.nan])))

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive.name.upper())


class TestAutocorrelation(PrimitiveTestBase):
    primitive = Autocorrelation

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, 2, 3, 1, 3, 2])
        assert round(primitive_func(array), 3) == -0.598

    def test_with_lag(self):
        primitive_instance = self.primitive(lag=3)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, 2, 3, 1, 2, 3])
        assert round(primitive_func(array), 3) == 1.0

    def test_starts_with_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, 2, 3, 1, 3, 2])
        assert round(primitive_func(array), 3) == -0.818

    def test_ends_with_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, 2, 3, 1, 3, np.nan])
        assert round(primitive_func(array), 3) == -0.636

    def test_all_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan])
        assert pd.isna(primitive_func(array))

    def test_negative_lag(self):
        primitive_instance = self.primitive(lag=-3)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, 2, 3, 1, 2, 3])
        assert round(primitive_func(array), 3) == 1.0

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive.name.upper())


class TestCorrelation(PrimitiveTestBase):
    primitive = Correlation
    array_1 = pd.Series([1, 4, 6, 7, 10, 12, 11.5])
    array_2 = pd.Series([1, 5, 9, 7, 11, 9, 1])

    def test_default_corr(self):
        correlation_val = 0.382596278303975
        primitive_func = self.primitive().get_function()
        assert np.isclose(self.array_1.corr(self.array_2), correlation_val)
        assert np.isclose(primitive_func(self.array_1, self.array_2), correlation_val)
        assert np.isclose(
            primitive_func(self.array_1, self.array_2),
            primitive_func(self.array_2, self.array_1),
        )

    def test_all_nans(self):
        primitive_func = self.primitive().get_function()
        array_nans = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        assert pd.isna(primitive_func(array_nans, array_nans))
        array_1_nans = self.array_1.copy().append(pd.Series([np.nan, np.nan]))
        array_2 = self.array_2.copy().append(pd.Series([12, 14]))
        assert primitive_func(array_1_nans, array_2) == 0.382596278303975

    def test_method(self):
        method = "kendall"
        correlation_val = 0.3504383220252312
        primitive_func = self.primitive(method=method).get_function()
        assert np.isclose(
            self.array_1.corr(self.array_2, method=method),
            correlation_val,
        )
        assert np.isclose(primitive_func(self.array_1, self.array_2), correlation_val)

    def test_errors(self):
        error_message = (
            "Invalid method, valid methods are ['pearson', 'spearman', 'kendall']"
        )
        with raises(ValueError, match=error_message):
            self.primitive(method="invalid")
            self.primitive(method=5)

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(
            pd_es,
            aggregation,
            transform,
            self.primitive.name.upper(),
            max_depth=2,
        )


class TestDateFirstEvent(PrimitiveTestBase):
    primitive = DateFirstEvent

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series(
            [
                "2011-04-09 10:30:00",
                "2011-04-09 10:30:06",
                "2011-04-09 10:30:12",
                "2011-04-09 10:30:18",
            ],
            dtype="datetime64[ns]",
        )
        answer = pd.Timestamp("2011-04-09 10:30:00")
        given_answer = primitive_func(case)
        assert given_answer == answer

    def test_nat(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series(
            [
                pd.NaT,
                pd.NaT,
                "2011-04-09 10:30:12",
                "2011-04-09 10:30:18",
            ],
            dtype="datetime64[ns]",
        )
        answer = pd.Timestamp("2011-04-09 10:30:12")
        given_answer = primitive_func(case)
        assert given_answer == answer

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive.name.upper())

    def test_serialize(self, es):
        test_serialize(self.primitive, es, target_dataframe_name="sessions")
