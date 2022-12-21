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
def test_expanding_count_series(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).count()
    num_nans = gap + min_periods - 1
    expected[range(num_nans)] = np.nan
    primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(window_series_pd.index)
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
def test_expanding_count_date_range(window_date_range_pd, min_periods, gap):
    test = _apply_gap_for_expanding_primitives(gap=gap, x=window_date_range_pd)
    expected = test.expanding(min_periods=min_periods).count()
    num_nans = gap + min_periods - 1
    expected[range(num_nans)] = np.nan
    primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(window_date_range_pd)
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
def test_expanding_min(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).min().values
    primitive_instance = ExpandingMin(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
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
def test_expanding_max(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).max().values
    primitive_instance = ExpandingMax(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
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
def test_expanding_std(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).std().values
    primitive_instance = ExpandingSTD(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
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
def test_expanding_mean(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).mean().values
    primitive_instance = ExpandingMean(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
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
def test_expanding_trend(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).aggregate(calculate_trend).values
    primitive_instance = ExpandingTrend(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
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
    window_series_pd,
    primitive,
):
    error_msg = (
        "String offsets are not supported for the gap parameter in Expanding primitives"
    )
    with pytest.raises(TypeError, match=error_msg):
        primitive(gap="2H").get_function()(
            numeric=window_series_pd,
            datetime=window_series_pd.index,
        )


def test_apply_gap_for_expanding_primitives_throws_error_when_given_string_offset(
    window_series_pd,
):
    error_msg = (
        "String offsets are not supported for the gap parameter in Expanding primitives"
    )
    with pytest.raises(TypeError, match=error_msg):
        _apply_gap_for_expanding_primitives(window_series_pd, gap="2H")


@pytest.mark.parametrize(
    "gap",
    [
        2,
        5,
        3,
        0,
    ],
)
def test_apply_gap_for_expanding_primitives(window_series_pd, gap):
    actual = _apply_gap_for_expanding_primitives(window_series_pd, gap).values
    expected = window_series_pd.shift(gap).values
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
    window_date_range_pd,
    gap,
):
    actual = pd.Series(
        _apply_gap_for_expanding_primitives(window_date_range_pd, gap).values,
    )
    expected = pd.Series(window_date_range_pd.to_series().shift(gap).values)
    pd.testing.assert_series_equal(actual, expected)
