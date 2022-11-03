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

# @pytest.mark.parametrize(
#     "min_periods, gap",
#     [
#         (5, 2),
#         (5, 0),
#     ],
# )
# def test_expanding_count(window_series_pd, min_periods, gap):
#     test = window_series_pd.shift(gap)
#     primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
#     actual = primitive_instance(window_series_pd.index)
#     expected = test.expanding(min_periods=min_periods).count().index
#     pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_min(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    primitive_instance = ExpandingMin(min_periods=min_periods, gap=gap).get_function()
    expected = pd.Series(test.expanding(min_periods=min_periods).min().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_max(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    primitive_instance = ExpandingMax(min_periods=min_periods, gap=gap).get_function()
    expected = pd.Series(test.expanding(min_periods=min_periods).max().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_std(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    primitive_instance = ExpandingSTD(min_periods=min_periods, gap=gap).get_function()
    expected = pd.Series(
        test.expanding(
            min_periods=min_periods,
        )
        .std()
        .values,
    )
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_mean(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    primitive_instance = ExpandingMean(min_periods=min_periods, gap=gap).get_function()
    expected = pd.Series(test.expanding(min_periods=min_periods).mean().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


# def test_expanding_trend(window_series_pd):
#     primitive_instance = ExpandingTrend().get_function()
#     expected = window_series_pd.expanding().trend()
#     actual = primitive_instance(numeric=window_series_pd, datetime=window_series_pd.index)
#     pd.testing.assert_series_equal(pd.Series(actual), expected)
