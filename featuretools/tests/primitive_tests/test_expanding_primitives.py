import numpy as np
import pandas as pd

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
    ExpandingMin,
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


def test_expanding_min():
    times = pd.date_range(start="2022-01-01", end="2023-01-01", periods=10)
    values = [i for i in range(10, 0)]
    series = pd.Series(data=values, index=times)
    primitive_instance = ExpandingMin().get_function()
    expected = pd.Series([i for i in range(10, 0)]).astype("float64")
    pd.testing.assert_series_equal(primitive_instance(series), expected)
