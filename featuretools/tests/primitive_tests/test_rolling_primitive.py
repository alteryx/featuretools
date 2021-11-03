import numpy as np
import pandas as pd
import pytest

# --> just add to primitives import
from featuretools.primitives.standard.rolling_transform_primitive import RollingMax
from featuretools.primitives.utils import roll_series_with_gap


def test_rolling_max_defaults():
    pass


def test_regular(rolling_series_pd):
    window_length = 5
    gap = 2
    min_periods = 5

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=min_periods).max().values

    primitive_instance = RollingMax(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


# def test_nan():
#     datetime = pd.date_range(
#         start='2019-01-01',
#         freq='1min',
#         name='datetime',
#         periods=5,
#     ).to_series().reset_index(drop=True)
#     numeric = pd.Series(range(5), name='numeric', dtype='float')
#     numeric.iloc[[0, 3]] = np.nan
#     primitive_instance = self.primitive(time_frame='1h')
#     primitive_func = primitive_instance.get_function()
#     given_answer = pd.Series(primitive_func(datetime, numeric))
#     answer = pd.Series([np.nan, 1, 1, 1, 1], dtype='float')
#     pd.testing.assert_series_equal(given_answer, answer)
