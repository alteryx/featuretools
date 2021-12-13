import pandas as pd
import pytest

from featuretools.primitives import (
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingSTD
)
from featuretools.primitives.utils import _roll_series_with_gap
from featuretools.tests.primitive_tests.utils import get_number_of_days


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ('6d', '7d'),
    ]
)
def test_rolling_max(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length,
                                          gap=gap,
                                          min_periods=window_length_num).max().values

    primitive_instance = RollingMax(window_length=window_length, gap=gap, min_periods=window_length_num)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap_num + window_length_num - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ('6d', '7d'),
    ]
)
def test_rolling_min(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length,
                                          gap=gap,
                                          min_periods=window_length_num).min().values

    primitive_instance = RollingMin(window_length=window_length, gap=gap, min_periods=window_length_num)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap_num + window_length_num - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ('6d', '7d'),
    ]
)
def test_rolling_mean(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length,
                                          gap=gap,
                                          min_periods=window_length_num).mean().values

    primitive_instance = RollingMean(window_length=window_length, gap=gap, min_periods=window_length_num)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap_num + window_length_num - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ('6d', '7d'),
    ]
)
def test_rolling_std(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length,
                                          gap=gap,
                                          min_periods=window_length_num).std().values

    primitive_instance = RollingSTD(window_length=window_length, gap=gap, min_periods=window_length_num)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    assert actual_vals.isna().sum() == gap_num + window_length_num - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ('6d', '7d'),
    ]
)
def test_rolling_count(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length,
                                          gap=gap,
                                          min_periods=window_length_num).count().values

    primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=window_length_num)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index))

    num_nans = gap_num + window_length_num - 1
    assert actual_vals.isna().sum() == num_nans
    # RollingCount will not match the exact _roll_series_with_gap call,
    # because it handles the min_periods difference within the primitive
    pd.testing.assert_series_equal(pd.Series(expected_vals).iloc[num_nans:], actual_vals.iloc[num_nans:])


@pytest.mark.parametrize(
    "min_periods, expected_num_nams",
    [
        (0, 2),  # 0 and 1 get treated the same
        (1, 2),
        (3, 4),
        (5, 6)
    ]
)
def test_rolling_count_primitive_min_periods_nans(min_periods, expected_num_nams, rolling_series_pd):
    window_length = '5d'
    gap = '2d'

    primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(rolling_series_pd.index))

    assert vals.isna().sum() == expected_num_nams


@pytest.mark.parametrize(
    "min_periods, expected_num_nams",
    [
        (0, 0),  # 0 and 1 get treated the same
        (1, 0),
        (3, 2),
        (5, 4)
    ]
)
def test_rolling_count_with_no_gap(min_periods, expected_num_nams, rolling_series_pd):
    window_length = '5d'
    gap = '0d'

    primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(rolling_series_pd.index))

    assert vals.isna().sum() == expected_num_nams
