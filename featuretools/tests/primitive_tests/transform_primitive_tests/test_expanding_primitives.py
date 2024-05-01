import numpy as np
import pandas as pd
import pytest

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingSTD,
    ExpandingTrend,
)
from featuretools.primitives.standard.transform.time_series.utils import (
    _apply_gap_for_expanding_primitives,
)
from featuretools.utils import calculate_trend


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
    ],
)
def test_expanding_count_series(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).count()
    num_nans = gap + min_periods - 1
    expected[range(num_nans)] = np.nan
    primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(window_series.index)
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_count_date_range(window_date_range, min_periods, gap):
    test = _apply_gap_for_expanding_primitives(gap=gap, x=window_date_range)
    expected = test.expanding(min_periods=min_periods).count()
    num_nans = gap + min_periods - 1
    expected[range(num_nans)] = np.nan
    primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(window_date_range)
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_min(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).min().values
    primitive_instance = ExpandingMin(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series,
        datetime=window_series.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_max(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).max().values
    primitive_instance = ExpandingMax(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series,
        datetime=window_series.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_std(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).std().values
    primitive_instance = ExpandingSTD(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series,
        datetime=window_series.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_mean(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).mean().values
    primitive_instance = ExpandingMean(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series,
        datetime=window_series.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
        (0, 0),
        (0, 1),
    ],
)
def test_expanding_trend(window_series, min_periods, gap):
    test = window_series.shift(gap)
    expected = test.expanding(min_periods=min_periods).aggregate(calculate_trend).values
    primitive_instance = ExpandingTrend(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series,
        datetime=window_series.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "primitive",
    [
        ExpandingMax,
        ExpandingMean,
        ExpandingMin,
        ExpandingSTD,
        ExpandingTrend,
    ],
)
def test_expanding_primitives_throw_error_when_given_string_offset(
    window_series,
    primitive,
):
    error_msg = (
        "String offsets are not supported for the gap parameter in Expanding primitives"
    )
    with pytest.raises(TypeError, match=error_msg):
        primitive(gap="2H").get_function()(
            numeric=window_series,
            datetime=window_series.index,
        )


def test_apply_gap_for_expanding_primitives_throws_error_when_given_string_offset(
    window_series,
):
    error_msg = (
        "String offsets are not supported for the gap parameter in Expanding primitives"
    )
    with pytest.raises(TypeError, match=error_msg):
        _apply_gap_for_expanding_primitives(window_series, gap="2H")


@pytest.mark.parametrize(
    "gap",
    [
        2,
        5,
        3,
        0,
    ],
)
def test_apply_gap_for_expanding_primitives(window_series, gap):
    actual = _apply_gap_for_expanding_primitives(window_series, gap).values
    expected = window_series.shift(gap).values
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "gap",
    [
        2,
        5,
        3,
        0,
    ],
)
def test_apply_gap_for_expanding_primitives_handles_date_range(
    window_date_range,
    gap,
):
    actual = pd.Series(
        _apply_gap_for_expanding_primitives(window_date_range, gap).values,
    )
    expected = pd.Series(window_date_range.to_series().shift(gap).values)
    pd.testing.assert_series_equal(actual, expected)
