import pandas as pd

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingSTD,
    ExpandingTrend,
)

"""
Test implementation.

TODO: Parametrize, refactor
"""


def test_expanding_count():
    times = pd.date_range(start="2022-01-01", end="2023-01-01", periods=50)
    primitive_instance = ExpandingCount().get_function()
    actual = primitive_instance(times)
    expected = pd.Series([i for i in range(0, 50)]).astype("float64")
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_expanding_min(window_series_pd):
    test = window_series_pd.shift(1)
    primitive_instance = ExpandingMin().get_function()
    expected = pd.Series(test.expanding(min_periods=1).min().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_expanding_max(window_series_pd):
    test = window_series_pd.shift(1)
    primitive_instance = ExpandingMax().get_function()
    expected = pd.Series(test.expanding(min_periods=1).max().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_expanding_std(window_series_pd):
    test = window_series_pd.shift(1)
    primitive_instance = ExpandingSTD().get_function()
    expected = pd.Series(test.expanding(min_periods=1).std().values)
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_expanding_mean(window_series_pd):
    test = window_series_pd.shift(1)
    primitive_instance = ExpandingMean().get_function()
    expected = pd.Series(test.expanding(min_periods=1).mean().values)
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
