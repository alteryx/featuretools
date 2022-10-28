import numpy as np
import pandas as pd

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
)

"""
Test implementation.

TODO: Parametrize, refactor
"""


def test_expanding_count(rolling_series_pd):
    times = pd.date_range(start="2022-01-01", end="2023-01-01", periods=50)
    primitive_instance = ExpandingCount().get_function()
    actual = primitive_instance(times)
    expected = pd.Series([i for i in range(0, 50)]).astype("float64")
    pd.testing.assert_series_equal(expected, pd.Series(actual))
