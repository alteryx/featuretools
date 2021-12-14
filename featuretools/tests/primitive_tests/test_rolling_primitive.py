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
        (5, 0),
        ('5d', '7d'),
        ('5d', '0d'),
    ]
)
@pytest.mark.parametrize(
    "min_periods",
    [1, 0, 2, 5]
)
def test_rolling_max(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length_num,
                                          gap=gap_num,
                                          min_periods=min_periods).max().values

    primitive_instance = RollingMax(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)

# --> test primitives with non uniformity


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ('5d', '7d'),
        ('5d', '0d'),
    ]
)
@pytest.mark.parametrize(
    "min_periods",
    [1, 0, 2, 5]
)
def test_rolling_min(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length_num,
                                          gap=gap_num,
                                          min_periods=min_periods).min().values

    primitive_instance = RollingMin(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ('5d', '7d'),
        ('5d', '0d'),
    ]
)
@pytest.mark.parametrize(
    "min_periods",
    [1, 0, 2, 5]
)
def test_rolling_mean(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length_num,
                                          gap=gap_num,
                                          min_periods=min_periods).mean().values

    primitive_instance = RollingMean(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ('5d', '7d'),
        ('5d', '0d'),
    ]
)
@pytest.mark.parametrize(
    "min_periods",
    [1, 0, 2, 5]
)
def test_rolling_std(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = _roll_series_with_gap(rolling_series_pd,
                                          window_length_num,
                                          gap=gap_num,
                                          min_periods=min_periods).std().values

    primitive_instance = RollingSTD(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)))

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 2

    if min_periods in [0, 1]:
        # the additional nan is because std pandas function returns NaN if there's only one value
        num_nans = gap_num + 1
    else:
        num_nans = gap_num + num_nans_from_min_periods - 1

    # The extra 1 at the beinning is because the std pandas function returns NaN if there's only one value
    assert actual_vals.isna().sum() == num_nans
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
                                          window_length_num,
                                          gap=gap_num,
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
        (1, 2),
        (3, 4),
        (5, 6)
    ]
)
@pytest.mark.parametrize("window_length, gap", [('5d', '2d'), (5, 2)])
def test_rolling_count_primitive_min_periods_nans(window_length, gap, min_periods, expected_num_nams, rolling_series_pd):
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
@pytest.mark.parametrize("window_length, gap", [('5d', '0d'), (5, 0)])
def test_rolling_count_with_no_gap(window_length, gap, min_periods, expected_num_nams, rolling_series_pd):
    primitive_instance = RollingCount(window_length=window_length, gap=gap, min_periods=min_periods)
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(rolling_series_pd.index))

    assert vals.isna().sum() == expected_num_nams
