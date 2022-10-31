import numpy as np
import pandas as pd

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
    ExpandingMean,
    ExpandingMin,
    ExpandingSTD,
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
    pd.testing.assert_series_equal(expected, pd.Series(actual))


def test_expanding_min(window_series_pd):
    primitive_instance = ExpandingMin().get_function()
    expected = window_series_pd.expanding().min()
    actual = primitive_instance(window_series_pd, index=window_series_pd.index)
    pd.testing.assert_series_equal(actual, expected)


def test_expanding_max(window_series_pd):
    primitive_instance = ExpandingMin().get_function()
    expected = window_series_pd.expanding().max()
    actual = primitive_instance(window_series_pd, index=window_series_pd.index)
    pd.testing.assert_series_equal(actual, expected)


def test_expanding_std(window_series_pd):
    primitive_instance = ExpandingSTD().get_function()
    expected = window_series_pd.expanding().std()
    actual = primitive_instance(window_series_pd, index=window_series_pd.index)
    pd.testing.assert_series_equal(actual, expected)


def test_expanding_mean(window_series_pd):
    primitive_instance = ExpandingMean().get_function()
    expected = window_series_pd.expanding().mean()
    actual = primitive_instance(window_series_pd, index=window_series_pd.index)
    pd.testing.assert_series_equal(actual, expected)
