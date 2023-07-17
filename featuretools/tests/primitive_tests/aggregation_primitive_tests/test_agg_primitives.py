from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pytest import raises

from featuretools.primitives import (
    AverageCountPerUnique,
    DateFirstEvent,
    Entropy,
    FirstLastTimeDelta,
    HasNoDuplicates,
    IsMonotonicallyDecreasing,
    IsMonotonicallyIncreasing,
    Kurtosis,
    MaxCount,
    MaxMinDelta,
    MedianCount,
    MinCount,
    NMostCommon,
    NMostCommonFrequency,
    NumFalseSinceLastTrue,
    NumPeaks,
    NumTrueSinceLastFalse,
    NumZeroCrossings,
    NUniqueDays,
    NUniqueDaysOfCalendarYear,
    NUniqueDaysOfMonth,
    NUniqueMonths,
    NUniqueWeeks,
    PercentTrue,
    Trend,
    Variance,
    get_aggregation_primitives,
)
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    check_serialize,
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
        valid_dfs(es, aggregation, transform, self.primitive)


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
        valid_dfs(es, aggregation, transform, self.primitive)


class TestEntropy(PrimitiveTestBase):
    primitive = Entropy

    @pytest.mark.parametrize(
        "dtype",
        ["category", "object", "string"],
    )
    def test_regular(self, dtype):
        data = pd.Series([1, 2, 3, 2], dtype=dtype)
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isclose(given_answer, 1.03, atol=0.01)

    @pytest.mark.parametrize(
        "dtype",
        ["category", "object", "string"],
    )
    def test_empty(self, dtype):
        data = pd.Series([], dtype=dtype)
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert given_answer == 0.0

    @pytest.mark.parametrize(
        "dtype",
        ["category", "object", "string"],
    )
    def test_args(self, dtype):
        data = pd.Series([1, 2, 3, 2], dtype=dtype)
        if dtype == "string":
            data = pd.concat([data, pd.Series([pd.NA, pd.NA], dtype=dtype)])
        else:
            data = pd.concat([data, pd.Series([np.nan, np.nan], dtype=dtype)])
        primitive_func = self.primitive(dropna=True, base=2).get_function()
        given_answer = primitive_func(data)
        assert np.isclose(given_answer, 1.5, atol=0.001)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive, max_depth=2)


class TestKurtosis(PrimitiveTestBase):
    primitive = Kurtosis

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64"],
    )
    def test_regular(self, dtype):
        data = pd.Series([1, 2, 3, 4, 5], dtype=dtype)
        answer = -1.3
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isclose(answer, given_answer, atol=0.01)

        data = pd.Series([1, 2, 3, 4, 5, 6], dtype=dtype)
        answer = -1.26
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isclose(answer, given_answer, atol=0.01)

        data = pd.Series([x * x for x in list(range(100))], dtype=dtype)
        answer = -0.85
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert np.isclose(answer, given_answer, atol=0.01)

        if dtype == "float64":
            # Series contains floating point values - only check with float dtype
            data = pd.Series([sqrt(x) for x in list(range(100))], dtype=dtype)
            answer = -0.46
            primitive_func = self.primitive().get_function()
            given_answer = primitive_func(data)
            assert np.isclose(answer, given_answer, atol=0.01)

    def test_nan(self):
        data = pd.Series([np.nan, 5, 3], dtype="float64")
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert pd.isna(given_answer)

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64"],
    )
    def test_empty(self, dtype):
        data = pd.Series([], dtype=dtype)
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert pd.isna(given_answer)

    def test_inf(self):
        data = pd.Series([1, np.inf], dtype="float64")
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert pd.isna(given_answer)

        data = pd.Series([np.NINF, 1, np.inf], dtype="float64")
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(data)
        assert pd.isna(given_answer)

    def test_arg(self):
        data = pd.Series([1, 2, 3, 4, 5, np.nan, np.nan], dtype="float64")
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
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


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
        data = pd.Series([], dtype="int64")
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
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


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
        valid_dfs(es, aggregation, transform, self.primitive)


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
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNumPeaks(PrimitiveTestBase):
    primitive = NumPeaks

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_negative_and_positive_nums(self, dtype):
        get_peaks = self.primitive().get_function()
        assert (
            get_peaks(pd.Series([-5, 0, 10, 0, 10, -5, -4, -5, 10, 0], dtype=dtype))
            == 4
        )

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_plateu(self, dtype):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 2, 3, 3, 3, 3, 3, 2, 1], dtype=dtype)) == 1
        assert get_peaks(pd.Series([1, 2, 3, 3, 3, 4, 3, 3, 3, 2, 1], dtype=dtype)) == 1
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
                    dtype=dtype,
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
                    dtype=dtype,
                ),
            )
            == 3
        )

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_regular(self, dtype):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 7, 3, 8, 2, 3, 4, 3, 4, 2, 4], dtype=dtype)) == 4
        assert get_peaks(pd.Series([1, 2, 3, 2, 1], dtype=dtype)) == 1

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_no_peak(self, dtype):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([1, 2, 3], dtype=dtype)) == 0
        assert get_peaks(pd.Series([3, 2, 2, 2, 2, 1], dtype=dtype)) == 0

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_too_small_data(self, dtype):
        get_peaks = self.primitive().get_function()
        assert get_peaks(pd.Series([], dtype=dtype)) == 0
        assert get_peaks(pd.Series([1])) == 0
        assert get_peaks(pd.Series([1, 1])) == 0
        assert get_peaks(pd.Series([1, 2])) == 0
        assert get_peaks(pd.Series([2, 1])) == 0

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "float64", "Int64"],
    )
    def test_nans(self, dtype):
        get_peaks = self.primitive().get_function()
        array = pd.Series(
            [
                0,
                5,
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
            ],
            dtype=dtype,
        )
        if dtype == "float64":
            array = pd.concat([array, pd.Series([np.nan, np.nan])])
        elif dtype == "Int64":
            array = pd.concat([array, pd.Series([pd.NA, pd.NA])])
        array = array.astype(dtype=dtype)
        assert get_peaks(array) == 3

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


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

    def test_empty(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([], dtype="datetime64[ns]")
        given_answer = primitive_func(case)
        assert pd.isna(given_answer)

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)

    def test_serialize(self, es):
        check_serialize(self.primitive, es, target_dataframe_name="sessions")


class TestMinCount(PrimitiveTestBase):
    primitive = MinCount

    def test_nan(self):
        data = pd.Series([np.nan, np.nan, np.nan])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert pd.isna(answer)

    def test_inf(self):
        data = pd.Series([5, 10, 10, np.inf, np.inf, np.inf])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 1

    def test_regular(self):
        data = pd.Series([1, 2, 2, 2, 3, 4, 4, 4, 5])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 1

        data = pd.Series([2, 2, 2, 3, 4, 4, 4])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 3

    def test_skipna(self):
        data = pd.Series([1, 1, 2, 3, 4, 4, np.nan, 5])
        primitive_func = self.primitive(skipna=False).get_function()
        answer = primitive_func(data)
        assert pd.isna(answer)

    def test_ninf(self):
        data = pd.Series([np.NINF, np.NINF, np.nan])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 2

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestMaxCount(PrimitiveTestBase):
    primitive = MaxCount

    def test_nan(self):
        data = pd.Series([np.nan, np.nan, np.nan])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert pd.isna(answer)

    def test_inf(self):
        data = pd.Series([5, 10, 10, np.inf, np.inf, np.inf])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 3

    def test_regular(self):
        data = pd.Series([1, 1, 2, 3, 4, 4, 4, 5])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 1

        data = pd.Series([1, 1, 2, 3, 4, 4, 4])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 3

    def test_skipna(self):
        data = pd.Series([1, 1, 2, 3, 4, 4, np.nan, 5])
        primitive_func = self.primitive(skipna=False).get_function()
        answer = primitive_func(data)
        assert pd.isna(answer)

    def test_ninf(self):
        data = pd.Series([np.NINF, np.NINF, np.nan])
        primitive_func = self.primitive().get_function()
        answer = primitive_func(data)
        assert answer == 2

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestMaxMinDelta(PrimitiveTestBase):
    primitive = MaxMinDelta
    array = pd.Series([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])

    def test_max_min_delta(self):
        primitive_func = self.primitive().get_function()
        assert primitive_func(self.array) == 7.0

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        array_nans = pd.concat([self.array, pd.Series([np.nan])])
        assert primitive_func(array_nans) == 7.0
        primitive_func = self.primitive(skipna=False).get_function()
        array_nans = pd.concat([self.array, pd.Series([np.nan])])
        assert pd.isna(primitive_func(array_nans))

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestMedianCount(PrimitiveTestBase):
    primitive = MedianCount

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([1, 3, 5, 7])
        given_answer = primitive_func(case)
        assert given_answer == 0

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([1, 3, 4, 4, 4, 5, 7, np.nan, np.nan])
        given_answer = primitive_func(case)
        assert given_answer == 3
        primitive_func = self.primitive(skipna=False).get_function()
        given_answer = primitive_func(case)
        assert pd.isna(given_answer)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNMostCommonFrequency(PrimitiveTestBase):
    primitive = NMostCommonFrequency

    def test_regular(self):
        test_cases = [
            pd.Series([8, 7, 10, 10, 10, 3, 4, 5, 10, 8, 7]),
            pd.Series([7, 7, 7, 6, 6, 5, 4]),
            pd.Series([4, 5, 6, 6, 7, 7, 7]),
        ]

        answers = [
            pd.Series([4, 2, 2]),
            pd.Series([3, 2, 1]),
            pd.Series([3, 2, 1]),
        ]

        primtive_func = self.primitive(3).get_function()

        for case, answer in zip(test_cases, answers):
            given_answer = primtive_func(case)
            given_answer = given_answer.reset_index(drop=True)
            assert given_answer.equals(answer)

    def test_n_larger_than_len(self):
        test_cases = [
            pd.Series(["red", "red", "blue", "green"]),
            pd.Series(["red", "red", "red", "blue", "green"]),
            pd.Series(["red", "blue", "green", "orange"]),
        ]
        answers = [
            pd.Series([2, 1, 1, np.nan, np.nan]),
            pd.Series([3, 1, 1, np.nan, np.nan]),
            pd.Series([1, 1, 1, 1, np.nan]),
        ]

        primtive_func = self.primitive(5).get_function()
        for case, answer in zip(test_cases, answers):
            given_answer = primtive_func(case)
            given_answer = given_answer.reset_index(drop=True)
            assert given_answer.equals(answer)

    def test_skipna(self):
        array = pd.Series(["red", "red", "blue", "green", np.nan, np.nan])
        primtive_func = self.primitive(5, skipna=False).get_function()
        given_answer = primtive_func(array)
        given_answer = given_answer.reset_index(drop=True)
        answer = pd.Series([2, 2, 1, 1, np.nan])
        assert given_answer.equals(answer)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        aggregation.append(self.primitive(5))
        valid_dfs(
            es,
            aggregation,
            transform,
            self.primitive,
            target_dataframe_name="customers",
            multi_output=True,
        )

    def test_with_featuretools_args(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        aggregation.append(self.primitive(5, skipna=False))
        valid_dfs(
            es,
            aggregation,
            transform,
            self.primitive,
            target_dataframe_name="customers",
            multi_output=True,
        )

    def test_serialize(self, es):
        check_serialize(
            primitive=self.primitive,
            es=es,
            target_dataframe_name="customers",
        )


class TestNUniqueDays(PrimitiveTestBase):
    primitive = NUniqueDays

    def test_two_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2011-12-31"))
        assert primitive_func(array) == 365 * 2

    def test_leap_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2016-01-01", "2017-12-31"))
        assert primitive_func(array) == 365 * 2 + 1

    def test_ten_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2019-12-31"))
        assert primitive_func(array) == 365 * 10 + 1 + 1

    def test_distinct_dt(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                datetime(2019, 2, 21),
                datetime(2019, 2, 1, 1, 20, 0),
                datetime(2019, 2, 1, 1, 30, 0),
                datetime(2018, 2, 1),
                datetime(2019, 1, 1),
            ],
        )
        assert primitive_func(array) == 4

    def test_NaT(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2011-12-31"))
        NaT_array = pd.Series([pd.NaT] * 100)
        assert primitive_func(pd.concat([array, NaT_array])) == 365 * 2

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNUniqueDaysOfCalendarYear(PrimitiveTestBase):
    primitive = NUniqueDaysOfCalendarYear

    def test_two_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2011-12-31"))
        assert primitive_func(array) == 365

    def test_leap_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2016-01-01", "2017-12-31"))
        assert primitive_func(array) == 366

    def test_ten_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2019-12-31"))
        assert primitive_func(array) == 366

    def test_distinct_dt(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                datetime(2019, 2, 21),
                datetime(2019, 2, 1, 1, 20, 0),
                datetime(2019, 2, 1, 1, 30, 0),
                datetime(2018, 2, 1),
                datetime(2019, 1, 1),
            ],
        )
        assert primitive_func(array) == 3

    def test_NaT(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2011-12-31"))
        NaT_array = pd.Series([pd.NaT] * 100)
        assert primitive_func(pd.concat([array, NaT_array])) == 365

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNUniqueDaysOfMonth(PrimitiveTestBase):
    primitive = NUniqueDaysOfMonth

    def test_two_days(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2010-01-02"))
        assert primitive_func(array) == 2

    def test_one_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2010-12-31"))
        assert primitive_func(array) == 31

    def test_leap_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2016-01-01", "2017-12-31"))
        assert primitive_func(array) == 31

    def test_distinct_dt(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                datetime(2019, 2, 21),
                datetime(2019, 2, 1, 1, 20, 0),
                datetime(2019, 2, 1, 1, 30, 0),
                datetime(2018, 2, 1),
                datetime(2019, 1, 1),
            ],
        )
        assert primitive_func(array) == 2

    def test_NaT(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2010-12-31"))
        NaT_array = pd.Series([pd.NaT] * 100)
        assert primitive_func(pd.concat([array, NaT_array])) == 31

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNUniqueMonths(PrimitiveTestBase):
    primitive = NUniqueMonths

    def test_two_days(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2010-01-02"))
        assert primitive_func(array) == 1

    def test_ten_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2019-12-31"))
        assert primitive_func(array) == 12 * 10

    def test_distinct_dt(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                datetime(2019, 2, 21),
                datetime(2019, 2, 1, 1, 20, 0),
                datetime(2019, 2, 1, 1, 30, 0),
                datetime(2018, 2, 1),
                datetime(2019, 1, 1),
            ],
        )
        assert primitive_func(array) == 3

    def test_NaT(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2011-12-31"))
        NaT_array = pd.Series([pd.NaT] * 100)
        assert primitive_func(pd.concat([array, NaT_array])) == 12 * 2

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNUniqueWeeks(PrimitiveTestBase):
    primitive = NUniqueWeeks

    def test_same_week(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2019-01-01", "2019-01-02"))
        assert primitive_func(array) == 1

    def test_ten_years(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2010-01-01", "2019-12-31"))
        assert primitive_func(array) == 523

    def test_distinct_dt(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                datetime(2019, 2, 21),
                datetime(2019, 2, 1, 1, 20, 0),
                datetime(2019, 2, 1, 1, 30, 0),
                datetime(2018, 2, 2),
                datetime(2019, 2, 3, 1, 30, 0),
                datetime(2019, 1, 1),
            ],
        )
        assert primitive_func(array) == 4

    def test_NaT(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(pd.date_range("2019-01-01", "2019-01-02"))
        NaT_array = pd.Series([pd.NaT] * 100)
        assert primitive_func(pd.concat([array, NaT_array])) == 1

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        aggregation.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestHasNoDuplicates(PrimitiveTestBase):
    primitive = HasNoDuplicates

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        data = pd.Series([1, 1, 2])
        assert not primitive_func(data)
        assert isinstance(primitive_func(data), bool)

        data = pd.Series([1, 2, 3])
        assert primitive_func(data)
        assert isinstance(primitive_func(data), bool)

        data = pd.Series([1, 2, 4])
        assert primitive_func(data)
        assert isinstance(primitive_func(data), bool)

        data = pd.Series(["red", "blue", "orange"])
        assert primitive_func(data)
        assert isinstance(primitive_func(data), bool)

        data = pd.Series(["red", "blue", "red"])
        assert not primitive_func(data)

    def test_nan(self):
        primitive_func = self.primitive().get_function()
        data = pd.Series([np.nan, 1, 2, 3])
        assert primitive_func(data)
        assert isinstance(primitive_func(data), bool)

        data = pd.Series([np.nan, np.nan, 1])
        # drop both nans, so has 1 value
        assert primitive_func(data) is True
        assert isinstance(primitive_func(data), bool)

        primitive_func = self.primitive(skipna=False).get_function()
        data = pd.Series([np.nan, np.nan, 1])
        assert primitive_func(data) is False
        assert isinstance(primitive_func(data), bool)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(
            es,
            aggregation,
            transform,
            self.primitive,
            target_dataframe_name="customers",
            instance_ids=[0, 1, 2],
        )


class TestIsMonotonicallyDecreasing(PrimitiveTestBase):
    primitive = IsMonotonicallyDecreasing

    def test_monotonically_decreasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([9, 5, 3, 1, -1])
        assert primitive_func(case) is True

    def test_monotonically_increasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, 5, 9])
        assert primitive_func(case) is False

    def test_non_monotonic(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, 2, 5])
        assert primitive_func(case) is False

    def test_weakly_decreasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([9, 3, 3, 1, -1])
        assert primitive_func(case) is True

    def test_nan(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([9, 5, 3, np.nan, 1, -1])
        assert primitive_func(case) is True

        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, np.nan, 5, 9])
        assert primitive_func(case) is False

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestIsMonotonicallyIncreasing(PrimitiveTestBase):
    primitive = IsMonotonicallyIncreasing

    def test_monotonically_increasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, 5, 9])
        assert primitive_func(case) is True

    def test_monotonically_decreasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([9, 5, 3, 1, -1])
        assert primitive_func(case) is False

    def test_non_monotonic(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, 2, 5])
        assert primitive_func(case) is False

    def test_weakly_increasing(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, 3, 9])
        assert primitive_func(case) is True

    def test_nan(self):
        primitive_func = self.primitive().get_function()
        case = pd.Series([-1, 1, 3, np.nan, 5, 9])
        assert primitive_func(case) is True

        primitive_func = self.primitive().get_function()
        case = pd.Series([9, 5, 3, np.nan, 1, -1])
        assert primitive_func(case) is False

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        aggregation.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive)
