import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import (
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingOutlierCount,
    RollingSTD,
    RollingTrend,
)
from featuretools.primitives.standard.transform.time_series.utils import (
    apply_rolling_agg_to_series,
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
def test_rolling_max(min_periods, window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)
    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = apply_rolling_agg_to_series(
        window_series_pd,
        lambda x: x.max(),
        window_length_num,
        gap=gap_num,
        min_periods=min_periods,
    )

    primitive_instance = RollingMax(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(window_series_pd.index, pd.Series(window_series_pd.values)),
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
def test_rolling_min(min_periods, window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = apply_rolling_agg_to_series(
        window_series_pd,
        lambda x: x.min(),
        window_length_num,
        gap=gap_num,
        min_periods=min_periods,
    )

    primitive_instance = RollingMin(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(window_series_pd.index, pd.Series(window_series_pd.values)),
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
def test_rolling_mean(min_periods, window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = apply_rolling_agg_to_series(
        window_series_pd,
        np.mean,
        window_length_num,
        gap=gap_num,
        min_periods=min_periods,
    )

    primitive_instance = RollingMean(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(window_series_pd.index, pd.Series(window_series_pd.values)),
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
def test_rolling_std(min_periods, window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Since we're using a uniform series we can check correctness using numeric parameters
    expected_vals = apply_rolling_agg_to_series(
        window_series_pd,
        lambda x: x.std(),
        window_length_num,
        gap=gap_num,
        min_periods=min_periods,
    )

    primitive_instance = RollingSTD(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(window_series_pd.index, pd.Series(window_series_pd.values)),
    )

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 2

    if min_periods in [0, 1]:
        # the additional nan is because std pandas function returns NaN if there's only one value
        num_nans = gap_num + 1
    else:
        num_nans = gap_num + num_nans_from_min_periods - 1

    # The extra 1 at the beginning is because the std pandas function returns NaN if there's only one value
    assert actual_vals.isna().sum() == num_nans
    pd.testing.assert_series_equal(pd.Series(expected_vals), actual_vals)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        ("6d", "7d"),
    ],
)
def test_rolling_count(window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    expected_vals = apply_rolling_agg_to_series(
        window_series_pd,
        lambda x: x.count(),
        window_length_num,
        gap=gap_num,
    )

    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=window_length_num,
    )
    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(primitive_func(window_series_pd.index))

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
    window_series_pd,
):
    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(window_series_pd.index))

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
    window_series_pd,
):
    primitive_instance = RollingCount(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )
    primitive_func = primitive_instance.get_function()
    vals = pd.Series(primitive_func(window_series_pd.index))

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
def test_rolling_trend(window_length, gap, expected_vals, window_series_pd):
    primitive_instance = RollingTrend(window_length=window_length, gap=gap)

    actual_vals = primitive_instance(window_series_pd.index, window_series_pd.values)

    pd.testing.assert_series_equal(pd.Series(expected_vals), pd.Series(actual_vals))


def test_rolling_trend_window_length_less_than_three(window_series_pd):
    primitive_instance = RollingTrend(window_length=2)

    vals = primitive_instance(window_series_pd.index, window_series_pd.values)

    for v in vals:
        assert np.isnan(v)


@pytest.mark.parametrize(
    "primitive",
    [
        RollingCount,
        RollingMax,
        RollingMin,
        RollingMean,
        RollingOutlierCount,
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


def test_rolling_trend_non_uniform():
    datetimes = (
        list(pd.date_range(start="2017-01-01", freq="1d", periods=3))
        + list(pd.date_range(start="2017-01-10", freq="2d", periods=4))
        + list(pd.date_range(start="2017-01-22", freq="1d", periods=7))
    )
    no_freq_series = pd.Series(range(len(datetimes)), index=datetimes)
    expected_series = pd.Series(
        [None, None, None]
        + [None, None, None, None]
        + [
            None,
            None,
            None,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )
    primitive_instance = RollingTrend(window_length="3d", gap="1d")
    rolled_series = pd.Series(
        primitive_instance(no_freq_series.index, pd.Series(no_freq_series.values)),
    )
    pd.testing.assert_series_equal(rolled_series, expected_series)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (5, 2),
        (5, 0),
        ("5d", "7d"),
        ("5d", "0d"),
    ],
)
@pytest.mark.parametrize(
    "min_periods",
    [1, 0, 2, 5],
)
def test_rolling_outlier_count(
    min_periods,
    window_length,
    gap,
    rolling_outlier_series_pd,
):
    primitive_instance = RollingOutlierCount(
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )

    primitive_func = primitive_instance.get_function()

    actual_vals = pd.Series(
        primitive_func(
            rolling_outlier_series_pd.index,
            pd.Series(rolling_outlier_series_pd.values),
        ),
    )

    expected_vals = apply_rolling_agg_to_series(
        series=rolling_outlier_series_pd,
        agg_func=primitive_instance.get_outliers_count,
        window_length=window_length,
        gap=gap,
        min_periods=min_periods,
    )

    # Since min_periods of 0 is the same as min_periods of 1
    num_nans_from_min_periods = min_periods or 1
    assert (
        actual_vals.isna().sum()
        == get_number_from_offset(gap) + num_nans_from_min_periods - 1
    )
    pd.testing.assert_series_equal(actual_vals, pd.Series(data=expected_vals))
