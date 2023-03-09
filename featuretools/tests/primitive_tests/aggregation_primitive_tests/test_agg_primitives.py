from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from pytest import raises

from featuretools.primitives import (
    AverageCountPerUnique,
    FirstLastTimeDelta,
    Kurtosis,
    NMostCommon,
    NumFalseSinceLastTrue,
    NumPeaks,
    NumTrueSinceLastFalse,
    NumZeroCrossings,
    PercentTrue,
    Trend,
    Variance,
    get_aggregation_primitives,
)
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
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

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


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

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


class TestKurtosis(PrimitiveTestBase):
    primitive = Kurtosis

    def test_nan(self):
        data = pd.Series([np.nan, 5, 3])
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isnan(given_answer)

    def test_empty(self):
        data = pd.Series([])
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isnan(given_answer)

    def test_inf(self):
        data = pd.Series([1, np.inf])
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isnan(given_answer)

        data = pd.Series([np.NINF, 1, np.inf])
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isnan(given_answer)

    def test_regular(self):
        data = pd.Series([1, 2, 3, 4, 5])
        answer = -1.3
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert answer == given_answer

        data = pd.Series([1, 2, 3, 4, 5, 6])
        answer = -1.2685714285714282
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert answer == given_answer

        data = pd.Series([x * x for x in list(range(100))])
        answer = -0.8516897715415088
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert answer == given_answer

        data = pd.Series([sqrt(x) for x in list(range(100))])
        answer = -0.4643347840875198
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert answer == given_answer

    def test_arg(self):
        data = pd.Series([1, 2, 3, 4, 5, np.nan, np.nan])
        answer = -1.3
        primitive_func = self.primitive(nan_policy="omit").get_function()
        given_answer = primitive_func(data)
        assert answer == given_answer

        primitive_func = self.primitive(nan_policy="propagate").get_function()
        given_answer = primitive_func(data)
        assert np.isnan(given_answer)

        primitive_func = self.primitive(nan_policy="raise").get_function()
        with raises(ValueError):
            primitive_func(data)

    def test_error(self):
        with raises(ValueError):
            self.primitive(nan_policy="invalid_policy").get_function()

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


class TestNumZeroCrossings(PrimitiveTestBase):
    primitive = NumZeroCrossings

    def test_nan(self):
        data = pd.Series([3, np.nan, 5, 3, np.nan, 0, np.nan, 0, np.nan, -2])
        # crossing from 0 to np.nan to -2, which is 1 crossing
        answer = 1
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

    def test_empty(self):
        data = pd.Series([])
        answer = 0
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

    def test_inf(self):
        data = pd.Series([-1, np.inf])
        answer = 1
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

        data = pd.Series([np.NINF, 1, np.inf])
        answer = 1
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

    def test_zeros(self):
        data = pd.Series([1, 0, -1, 0, 1, 0, -1])
        answer = 3
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

        data = pd.Series([1, 0, 1, 0, 1])
        answer = 0
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

    def test_regular(self):
        data = pd.Series([1, 2, 3, 4, 5])
        answer = 0
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

        data = pd.Series([1, -1, 2, -2, 3, -3])
        answer = 5
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        assert given_answer == answer

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


class TestNumTrueSinceLastFalse(PrimitiveTestBase):
    primitive = NumTrueSinceLastFalse

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([False, True, False, True, True])
        answer = primitive_func(bools)
        correct_answer = 2
        assert answer == correct_answer

    def test_regular_end_in_false(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([False, True, False, True, True, False])
        answer = primitive_func(bools)
        correct_answer = 0
        assert answer == correct_answer

    def test_no_false(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True] * 5)
        assert pd.isna(primitive_func(bools))

    def test_all_false(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([False, False, False])
        answer = primitive_func(bools)
        correct_answer = 0
        assert answer == correct_answer

    def test_nan(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([False, True, np.nan, True, True])
        answer = primitive_func(bools)
        correct_answer = 3
        assert answer == correct_answer

    def test_all_nan(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([np.nan, np.nan, np.nan])
        assert pd.isna(primitive_func(bools))

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


class TestNumFalseSinceLastTrue(PrimitiveTestBase):
    primitive = NumFalseSinceLastTrue

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True, False, True, False, False])
        answer = primitive_func(bools)
        correct_answer = 2
        assert answer == correct_answer

    def test_regular_end_in_true(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True, False, True, False, False, True])
        answer = primitive_func(bools)
        correct_answer = 0
        assert answer == correct_answer

    def test_no_true(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([False] * 5)
        assert pd.isna(primitive_func(bools))

    def test_all_true(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True, True, True])
        answer = primitive_func(bools)
        correct_answer = 0
        assert answer == correct_answer

    def test_nan(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True, False, np.nan, False, False])
        answer = primitive_func(bools)
        correct_answer = 3
        assert answer == correct_answer

    def test_all_nan(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([np.nan, np.nan, np.nan])
        assert pd.isna(primitive_func(bools))

    def test_numeric_and_string_input(self):
        primitive_func = self.primitive().get_function()
        bools = pd.Series([True, 0, 1, "10", ""])
        answer = primitive_func(bools)
        correct_answer = 1
        assert answer == correct_answer

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())


class TestNumPeaks(PrimitiveTestBase):
    primitive = NumPeaks

    def test_negative_and_positive_nums(self):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([-5, 0, 10, 0, 10, -5, -4, -5, 10, 0])) == 4

    def test_plateu(self):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 2, 3, 3, 3, 3, 3, 2, 1])) == 1
        assert get_peaks(pd.Series([1, 2, 3, 3, 3, 4, 3, 3, 3, 2, 1])) == 1
        assert (
            get_peaks(
                pd.Series(
                    [
                        5,
                        4,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        4,
                        5,
                        5,
                        5,
                        5,
                        5,
                        3,
                        3,
                        3,
                        3,
                        4,
                    ],
                ),
            )
            == 1
        )
        assert (
            get_peaks(
                pd.Series(
                    [
                        1,
                        2,
                        3,
                        3,
                        3,
                        2,
                        1,
                        2,
                        3,
                        3,
                        3,
                        2,
                        5,
                        5,
                        5,
                        2,
                    ],
                ),
            )
            == 3
        )

    def test_regular(self):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 7, 3, 8, 2, 3, 4, 3, 4, 2, 4])) == 4
        assert get_peaks(pd.Series([1, 2, 3, 2, 1])) == 1

    def test_no_peak(self):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 2, 3])) == 0
        assert get_peaks(pd.Series([3, 2, 2, 2, 2, 1])) == 0

    def test_too_small_data(self):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([])) == 0
        assert get_peaks(pd.Series([1])) == 0
        assert get_peaks(pd.Series([1, 1])) == 0
        assert get_peaks(pd.Series([1, 2])) == 0
        assert get_peaks(pd.Series([2, 1])) == 0

    def test_nans(self):
        get_peaks = self.primitive().get_function()
        array = pd.Series(
            [
                0.0,
                5.0,
                10,
                15,
                20,
                0,
                1,
                2,
                3,
                0,
                0,
                5,
                0,
                7,
                14,
                np.NaN,
                np.NaN,
            ],
        )
        assert get_peaks(array) == 3

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())
