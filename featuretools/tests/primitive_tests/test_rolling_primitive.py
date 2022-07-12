from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np

from featuretools.primitives import (
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingSTD,
    RollingTrend,
    roll_series_with_gap,
)
from featuretools.tests.primitive_tests.utils import get_number_from_offset


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ("5d", "7d"),
        ("5d", "0d"),
    ],
)
@pytest.mark.parametrize("min_periods", [1, 0, 2, 5])
def test_rolling_max(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)
    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = (
        roll_series_with_gap(
            rolling_series_pd,
            window_length_num,
            gap=gap_num,
            min_periods=min_periods,
        )
        .max()
        .values
    )

    primitive_instance = RollingMax(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)),
    )

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ("5d", "7d"),
        ("5d", "0d"),
    ],
)
@pytest.mark.parametrize("min_periods", [1, 0, 2, 5])
def test_rolling_min(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = (
        roll_series_with_gap(
            rolling_series_pd,
            window_length_num,
            gap=gap_num,
            min_periods=min_periods,
        )
        .min()
        .values
    )

    primitive_instance = RollingMin(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)),
    )

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ("5d", "7d"),
        ("5d", "0d"),
    ],
)
@pytest.mark.parametrize("min_periods", [1, 0, 2, 5])
def test_rolling_mean(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = (
        roll_series_with_gap(
            rolling_series_pd,
            window_length_num,
            gap=gap_num,
            min_periods=min_periods,
        )
        .mean()
        .values
    )

    primitive_instance = RollingMean(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)),
    )

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1

    assert actual_vals.isna().sum() == gap_num + num_nans_from_min_periods - 1
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ("5d", "7d"),
        ("5d", "0d"),
    ],
)
@pytest.mark.parametrize("min_periods", [1, 0, 2, 5])
def test_rolling_std(min_periods, window_length, gap, rolling_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = (
        roll_series_with_gap(
            rolling_series_pd,
            window_length_num,
            gap=gap_num,
            min_periods=min_periods,
        )
        .std()
        .values
    )

    primitive_instance = RollingSTD(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(rolling_series_pd.index, pd.Series(rolling_series_pd.values)),
    )

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
        ("6d", "7d"),
    ],
)
def test_rolling_count(window_length, gap, rolling_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    expected_vals = (
        roll_series_with_gap(
            rolling_series_pd,
            window_length_num,
            gap=gap_num,
            min_periods=window_length_num,
        )
        .count()
        .values
    )

    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=window_length_num,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(rolling_series_pd.index))

    num_nans = gap_num + window_length_num - 1
    assert actual_vals.isna().sum() == num_nans
    # RollingCount will not match the exact roll_series_with_gap call,
    # because it handles the min_periods difference within the primitive
    pd.testing.assert_series_equal(
        pd.Series(expected_vals).iloc[num_nans:],
        actual_vals.iloc[num_nans:],
    )


@pytest.mark.parametrize(
    "min_periods, expected_num_nams",
    [(0, 2), (1, 2), (3, 4), (5, 6)],  # 0 and 1 get treated the same
)
@pytest.mark.parametrize("window_length, gap", [("5d", "2d"), (5, 2)])
def test_rolling_count_primitive_min_periods_nans(
    window_length,
    gap,
    min_periods,
    expected_num_nams,
    rolling_series_pd,
):
    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(rolling_series_pd.index))

    assert vals.isna().sum() == expected_num_nams


@pytest.mark.parametrize(
    "min_periods, expected_num_nams",
    [(0, 0), (1, 0), (3, 2), (5, 4)],  # 0 and 1 get treated the same
)
@pytest.mark.parametrize("window_length, gap", [("5d", "0d"), (5, 0)])
def test_rolling_count_with_no_gap(
    window_length,
    gap,
    min_periods,
    expected_num_nams,
    rolling_series_pd,
):
    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(rolling_series_pd.index))

    assert vals.isna().sum() == expected_num_nams


@pytest.mark.parametrize(
    "window_length, gap, expected_vals",
    [
        (3, 0, [np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (
            4,
            1,
            [np.nan, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            "5d",
            "7d",
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ),
        (
            "5d",
            "0d",
            [np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
    ],
)
def test_rolling_trend(window_length, gap, expected_vals, rolling_series_pd):
    primitive_instance = RollingTrend(window_length=window_length, gap=gap)

    actual_vals = primitive_instance(rolling_series_pd.index, rolling_series_pd.values)

    pd.testing.assert_series_equal(pd.Series(expected_vals), pd.Series(actual_vals))


def test_rolling_trend_window_length_less_than_three(rolling_series_pd):
    primitive_instance = RollingTrend(window_length=2)

    vals = primitive_instance(rolling_series_pd.index, rolling_series_pd.values)

    for v in vals:
        assert np.isnan(v)


@pytest.mark.parametrize(
    "min_periods,expected_vals",
    [
        (0, [np.nan, np.nan, 1.5, 2.3, 4.6, 6.8, 12.8, 26.4, 55.2, 110.4]),
        (
            2,
            [np.nan, np.nan, 1.5, 2.3, 4.6, 6.8, 12.8, 26.4, 55.2, 110.4],
        ),
        (
            3,
            [np.nan, np.nan, 1.5, 2.3, 4.6, 6.8, 12.8, 26.4, 55.2, 110.4],
        ),
    ],
)
def test_rolling_trend_min_periods(min_periods, expected_vals):
    times = pd.date_range(start="2019-01-01", freq="1D", periods=10)
    primitive_instance = RollingTrend(window_length=3, min_periods=min_periods)
    actual_vals = primitive_instance(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384])
    pd.testing.assert_series_equal(pd.Series(expected_vals), pd.Series(actual_vals))


@pytest.mark.parametrize(
    "primitive",
    [
        RollingCount,
        RollingMax,
        RollingMin,
        RollingMean,
    ],
)
def test_rolling_primitives_non_uniform(primitive):
    # When the data isn't uniform, this impacts the number of values in each rolling window
    datetimes = (
        list(pd.date_range(start="2017-01-01", freq="1d", periods=3))
        + list(pd.date_range(start="2017-01-10", freq="2d", periods=4))
        + list(pd.date_range(start="2017-01-22", freq="1d", periods=7))
    )
    no_freq_series = pd.Series(range(len(datetimes)), index=datetimes)

    # Should match RollingCount exactly and have same nan values as other primitives
    expected_series = pd.Series(
        [None, 1, 2] + [None, 1, 1, 1] + [None, 1, 2, 3, 3, 3, 3],
    )

    primitive_instance = primitive(window_length="3d", gap="1d")
    if isinstance(primitive_instance, RollingCount):
        rolled_series = pd.Series(primitive_instance(no_freq_series.index))
        pd.testing.assert_series_equal(rolled_series, expected_series)
    else:
        rolled_series = pd.Series(
            primitive_instance(no_freq_series.index, pd.Series(no_freq_series.values)),
        )
        pd.testing.assert_series_equal(expected_series.isna(), rolled_series.isna())


def test_rolling_std_non_uniform():
    # When the data isn't uniform, this impacts the number of values in each rolling window
    datetimes = (
        list(pd.date_range(start="2017-01-01", freq="1d", periods=3))
        + list(pd.date_range(start="2017-01-10", freq="2d", periods=4))
        + list(pd.date_range(start="2017-01-22", freq="1d", periods=7))
    )
    no_freq_series = pd.Series(range(len(datetimes)), index=datetimes)

    # There will be at least two null values at the beginning of each range's rows, the first for the
    # row skipped by the gap, and the second because pandas' std returns NaN if there's only one row
    expected_series = pd.Series(
        [None, None, 0.707107]
        + [None, None, None, None]
        + [  # Because the freq was 2 days, there will never be more than 1 observation
            None,
            None,
            0.707107,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    primitive_instance = RollingSTD(window_length="3d", gap="1d")
    rolled_series = pd.Series(
        primitive_instance(no_freq_series.index, pd.Series(no_freq_series.values)),
    )

    pd.testing.assert_series_equal(rolled_series, expected_series)


@pytest.mark.parametrize(
    "primitive",
    [RollingCount, RollingMax, RollingMin, RollingMean, RollingSTD, RollingTrend],
)
@patch(
    "featuretools.primitives.rolling_transform_primitive._apply_roll_with_offset_gap",
)
def test_no_call_to_apply_roll_with_offset_gap_with_numeric(
    mock_apply_roll,
    primitive,
    rolling_series_pd,
):
    assert not mock_apply_roll.called

    fully_numeric_primitive = primitive(window_length=3, gap=1)
    primitive_func = fully_numeric_primitive.get_function()
    if isinstance(fully_numeric_primitive, RollingCount):
        pd.Series(primitive_func(rolling_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                rolling_series_pd.index,
                pd.Series(rolling_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    offset_window_primitive = primitive(window_length="3d", gap=1)
    primitive_func = offset_window_primitive.get_function()
    if isinstance(offset_window_primitive, RollingCount):
        pd.Series(primitive_func(rolling_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                rolling_series_pd.index,
                pd.Series(rolling_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    no_gap_specified_primitive = primitive(window_length="3d")
    primitive_func = no_gap_specified_primitive.get_function()
    if isinstance(no_gap_specified_primitive, RollingCount):
        pd.Series(primitive_func(rolling_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                rolling_series_pd.index,
                pd.Series(rolling_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    no_gap_specified_primitive = primitive(window_length="3d", gap="1d")
    primitive_func = no_gap_specified_primitive.get_function()
    if isinstance(no_gap_specified_primitive, RollingCount):
        pd.Series(primitive_func(rolling_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                rolling_series_pd.index,
                pd.Series(rolling_series_pd.values),
            ),
        )

    assert mock_apply_roll.called
