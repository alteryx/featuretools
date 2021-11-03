import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import pytest

# --> just add to primitives import
from featuretools.primitives.standard.rolling_transform_primitive import RollingCount, RollingMax, RollingMin, RollingMean, RollingSTD
from featuretools.primitives.utils import roll_series_with_gap


def test_rolling_max(rolling_series_pd):
    window_length = 5
    gap = 2

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=window_length).max().values

    primitive_instance = RollingMax(window_length=window_length, gap=gap, min_periods=window_length)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap + window_length - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)
    # --> maybe test early values that they're as expected


def test_rolling_min(rolling_series_pd):
    window_length = 5
    gap = 2

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=window_length).min().values

    primitive_instance = RollingMin(window_length=window_length, gap=gap, min_periods=window_length)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap + window_length - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)
    # --> maybe test early values that they're as expected


def test_rolling_mean(rolling_series_pd):
    window_length = 5
    gap = 2

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=window_length).mean().values

    primitive_instance = RollingMean(window_length=window_length, gap=gap, min_periods=window_length)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap + window_length - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)
    # --> maybe test early values that they're as expected


def test_rolling_std(rolling_series_pd):
    window_length = 5
    gap = 2

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=window_length).std().values

    primitive_instance = RollingSTD(window_length=window_length, gap=gap, min_periods=window_length)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap + window_length - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)
    # --> maybe test early values that they're as expected


def test_rolling_count(rolling_series_pd):
    window_length = 5
    gap = 2

    expected_vals = roll_series_with_gap(rolling_series_pd,
                                         window_length,
                                         gap=gap,
                                         min_periods=window_length).count().values

    primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=window_length)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    # Count
    num_nans = gap + window_length - 1
    assert actual_vals.isna().sum() == num_nans
    pd.testing.assert_series_equal(pd.Series(expected_vals).iloc[num_nans:], actual_vals.iloc[num_nans:])


def test_rolling_count_primitive_min_periods_nans(rolling_series_pd):
    window_length = 5
    gap = 2

    for min_periods in range(window_length + 1):
        primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=min_periods)
        primitive_func = primitive_instance.get_function()
        vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

        if min_periods == 0:
            # when min periods is 0 it's treated the same as if it's 1
            num_nans = gap
        else:
            num_nans = gap + min_periods - 1
        assert vals.isna().sum() == num_nans


def test_rolling_count_with_no_gap(rolling_series_pd):
    window_length = 5
    gap = 0

    for min_periods in range(window_length + 1):
        primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=min_periods)
        primitive_func = primitive_instance.get_function()
        vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

        if min_periods == 0:
            # when min periods is 0 it's treated the same as if it's 1
            num_nans = gap
        else:
            num_nans = gap + min_periods - 1
        assert vals.isna().sum() == num_nans
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
