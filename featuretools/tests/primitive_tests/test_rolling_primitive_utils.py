from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import (
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingSTD,
    RollingTrend,
)
from featuretools.primitives.standard.transform.time_series.utils import (
    _get_rolled_series_without_gap,
    apply_roll_with_offset_gap,
    roll_series_with_gap,
)
from featuretools.tests.primitive_tests.utils import get_number_from_offset


def test_get_rolled_series_without_gap(window_series_pd):
    # Data is daily, so number of rows should be number of days not included in the gap
    assert len(_get_rolled_series_without_gap(window_series_pd, "11D")) == 9
    assert len(_get_rolled_series_without_gap(window_series_pd, "0D")) == 20
    assert len(_get_rolled_series_without_gap(window_series_pd, "48H")) == 18
    assert len(_get_rolled_series_without_gap(window_series_pd, "4H")) == 19


def test_get_rolled_series_without_gap_not_uniform(window_series_pd):
    non_uniform_series = window_series_pd.iloc[[0, 2, 5, 6, 8, 9]]

    assert len(_get_rolled_series_without_gap(non_uniform_series, "10D")) == 0
    assert len(_get_rolled_series_without_gap(non_uniform_series, "0D")) == 6
    assert len(_get_rolled_series_without_gap(non_uniform_series, "48H")) == 4
    assert len(_get_rolled_series_without_gap(non_uniform_series, "4H")) == 5
    assert len(_get_rolled_series_without_gap(non_uniform_series, "4D")) == 3
    assert len(_get_rolled_series_without_gap(non_uniform_series, "4D2H")) == 2


def test_get_rolled_series_without_gap_empty_series(window_series_pd):
    empty_series = pd.Series([], dtype="object")
    assert len(_get_rolled_series_without_gap(empty_series, "1D")) == 0
    assert len(_get_rolled_series_without_gap(empty_series, "0D")) == 0


def test_get_rolled_series_without_gap_large_bound(window_series_pd):
    assert len(_get_rolled_series_without_gap(window_series_pd, "100D")) == 0
    assert (
        len(
            _get_rolled_series_without_gap(
                window_series_pd.iloc[[0, 2, 5, 6, 8, 9]],
                "20D",
            ),
        )
        == 0
    )


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (3, 2),
        (3, 4),  # gap larger than window
        (2, 0),  # gap explicitly set to 0
        ("3d", "2d"),  # using offset aliases
        ("3d", "4d"),
        ("4d", "0d"),
    ],
)
def test_roll_series_with_gap(window_length, gap, window_series_pd):
    rolling_max = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=1,
    ).max()
    rolling_min = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=1,
    ).min()

    assert len(rolling_max) == len(window_series_pd)
    assert len(rolling_min) == len(window_series_pd)

    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)
    for i in range(len(window_series_pd)):
        start_idx = i - gap_num - window_length_num + 1

        if isinstance(gap, str):
            # No gap functionality is happening, so gap isn't taken account in the end index
            # it's like the gap is 0; it includes the row itself
            end_idx = i
        else:
            end_idx = i - gap_num

        # If start and end are negative, they're entirely before
        if start_idx < 0 and end_idx < 0:
            assert pd.isnull(rolling_max.iloc[i])
            assert pd.isnull(rolling_min.iloc[i])
            continue

        if start_idx < 0:
            start_idx = 0

        # Because the row values are a range from 0 to 20, the rolling min will be the start index
        # and the rolling max will be the end idx
        assert rolling_min.iloc[i] == start_idx
        assert rolling_max.iloc[i] == end_idx


@pytest.mark.parametrize("window_length", [3, "3d"])
def test_roll_series_with_no_gap(window_length, window_series_pd):
    actual_rolling = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=0,
        min_periods=1,
    ).mean()
    expected_rolling = window_series_pd.rolling(window_length, min_periods=1).mean()

    pd.testing.assert_series_equal(actual_rolling, expected_rolling)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (6, 2),
        (6, 0),  # No gap - changes early values
        ("6d", "0d"),  # Uses offset aliases
        ("6d", "2d"),
    ],
)
def test_roll_series_with_gap_early_values(window_length, gap, window_series_pd):
    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)

    # Default min periods is 1 - will include all
    default_partial_values = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=1,
    ).count()
    num_empty_aggregates = len(default_partial_values.loc[default_partial_values == 0])
    num_partial_aggregates = len(
        (default_partial_values.loc[default_partial_values != 0]).loc[
            default_partial_values < window_length_num
        ],
    )

    assert num_partial_aggregates == window_length_num - 1
    if isinstance(gap, str):
        # gap isn't handled, so we'll always at least include the row itself
        assert num_empty_aggregates == 0
    else:
        assert num_empty_aggregates == gap_num

    # Make min periods the size of the window
    no_partial_values = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=window_length_num,
    ).count()
    num_null_aggregates = len(no_partial_values.loc[pd.isna(no_partial_values)])
    num_partial_aggregates = len(
        no_partial_values.loc[no_partial_values < window_length_num],
    )

    # because we shift, gap is included as nan values in the series.
    # Count treats nans in a window as values that don't get counted,
    # so the gap rows get included in the count for whether a window has "min periods".
    # This is different than max, for example, which does not count nans in a window as values towards "min periods"
    assert num_null_aggregates == window_length_num - 1
    if isinstance(gap, str):
        # gap isn't handled, so we'll never have any partial aggregates
        assert num_partial_aggregates == 0
    else:
        assert num_partial_aggregates == gap_num


def test_roll_series_with_gap_nullable_types(window_series_pd):
    window_length = 3
    gap = 2
    min_periods = 1
    # Because we're inserting nans, confirm that nullability of the dtype doesn't have an impact on the results
    nullable_series = window_series_pd.astype("Int64")
    non_nullable_series = window_series_pd.astype("int64")

    nullable_rolling_max = roll_series_with_gap(
        nullable_series,
        window_length,
        gap=gap,
        min_periods=min_periods,
    ).max()
    non_nullable_rolling_max = roll_series_with_gap(
        non_nullable_series,
        window_length,
        gap=gap,
        min_periods=min_periods,
    ).max()

    pd.testing.assert_series_equal(nullable_rolling_max, non_nullable_rolling_max)


def test_roll_series_with_gap_nullable_types_with_nans(window_series_pd):
    window_length = 3
    gap = 2
    min_periods = 1
    nullable_floats = window_series_pd.astype("float64").replace(
        {1: np.nan, 3: np.nan},
    )
    nullable_ints = nullable_floats.astype("Int64")

    nullable_ints_rolling_max = roll_series_with_gap(
        nullable_ints,
        window_length,
        gap=gap,
        min_periods=min_periods,
    ).max()
    nullable_floats_rolling_max = roll_series_with_gap(
        nullable_floats,
        window_length,
        gap=gap,
        min_periods=min_periods,
    ).max()

    pd.testing.assert_series_equal(
        nullable_ints_rolling_max,
        nullable_floats_rolling_max,
    )

    expected_early_values = [np.nan, np.nan, 0, 0, 2, 2, 4] + list(
        range(7 - gap, len(window_series_pd) - gap),
    )
    for i in range(len(window_series_pd)):
        actual = nullable_floats_rolling_max.iloc[i]
        expected = expected_early_values[i]

        if pd.isnull(actual):
            assert pd.isnull(expected)
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "window_length, gap",
    [
        ("3d", "2d"),
        ("3d", "4d"),
        ("4d", "0d"),
    ],
)
def test_apply_roll_with_offset_gap(window_length, gap, window_series_pd):
    def max_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, max, min_periods=1)

    rolling_max_obj = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=1,
    )
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    def min_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, min, min_periods=1)

    rolling_min_obj = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=1,
    )
    rolling_min_series = rolling_min_obj.apply(min_wrapper)

    assert len(rolling_max_series) == len(window_series_pd)
    assert len(rolling_min_series) == len(window_series_pd)

    gap_num = get_number_from_offset(gap)
    window_length_num = get_number_from_offset(window_length)
    for i in range(len(window_series_pd)):
        start_idx = i - gap_num - window_length_num + 1
        # Now that we have the _apply call, this acts as expected
        end_idx = i - gap_num

        # If start and end are negative, they're entirely before
        if start_idx < 0 and end_idx < 0:
            assert pd.isnull(rolling_max_series.iloc[i])
            assert pd.isnull(rolling_min_series.iloc[i])
            continue

        if start_idx < 0:
            start_idx = 0

        # Because the row values are a range from 0 to 20, the rolling min will be the start index
        # and the rolling max will be the end idx
        assert rolling_min_series.iloc[i] == start_idx
        assert rolling_max_series.iloc[i] == end_idx


@pytest.mark.parametrize(
    "min_periods",
    [1, 0, None],
)
def test_apply_roll_with_offset_gap_default_min_periods(min_periods, window_series_pd):
    window_length = "5d"
    window_length_num = 5
    gap = "3d"
    gap_num = 3

    def count_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, len, min_periods=min_periods)

    rolling_count_obj = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=min_periods,
    )
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    # gap essentially creates a rolling series that has no elements; which should be nan
    # to differentiate from when a window only has null values
    num_empty_aggregates = rolling_count_series.isna().sum()
    num_partial_aggregates = len(
        (rolling_count_series.loc[rolling_count_series != 0]).loc[
            rolling_count_series < window_length_num
        ],
    )

    assert num_empty_aggregates == gap_num
    assert num_partial_aggregates == window_length_num - 1


@pytest.mark.parametrize(
    "min_periods",
    [2, 3, 4, 5],
)
def test_apply_roll_with_offset_gap_min_periods(min_periods, window_series_pd):
    window_length = "5d"
    window_length_num = 5
    gap = "3d"
    gap_num = 3

    def count_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, len, min_periods=min_periods)

    rolling_count_obj = roll_series_with_gap(
        window_series_pd,
        window_length,
        gap=gap,
        min_periods=min_periods,
    )
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    # gap essentially creates rolling series that have no elements; which should be nan
    # to differentiate from when a window only has null values
    num_empty_aggregates = rolling_count_series.isna().sum()
    num_partial_aggregates = len(
        (rolling_count_series.loc[rolling_count_series != 0]).loc[
            rolling_count_series < window_length_num
        ],
    )

    assert num_empty_aggregates == min_periods - 1 + gap_num
    assert num_partial_aggregates == window_length_num - min_periods


def test_apply_roll_with_offset_gap_non_uniform():
    window_length = "3d"
    gap = "3d"
    min_periods = 1
    # When the data isn't uniform, this impacts the number of values in each rolling window
    datetimes = (
        list(pd.date_range(start="2017-01-01", freq="1d", periods=7))
        + list(pd.date_range(start="2017-02-01", freq="2d", periods=7))
        + list(pd.date_range(start="2017-03-01", freq="1d", periods=7))
    )
    no_freq_series = pd.Series(range(len(datetimes)), index=datetimes)

    assert pd.infer_freq(no_freq_series.index) is None

    expected_series = pd.Series(
        [None, None, None, 1, 2, 3, 3]
        + [None, None, 1, 1, 1, 1, 1]
        + [None, None, None, 1, 2, 3, 3],
        index=datetimes,
    )

    def count_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, len, min_periods=min_periods)

    rolling_count_obj = roll_series_with_gap(
        no_freq_series,
        window_length,
        gap=gap,
        min_periods=min_periods,
    )
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    pd.testing.assert_series_equal(rolling_count_series, expected_series)


def test_apply_roll_with_offset_data_frequency_higher_than_parameters_frequency():
    window_length = "5D"  # 120 hours
    window_length_num = 5
    # In order for min periods to be the length of the window, we multiply 24hours*5
    min_periods = window_length_num * 24

    datetimes = list(pd.date_range(start="2017-01-01", freq="1H", periods=200))
    high_frequency_series = pd.Series(range(200), index=datetimes)

    # Check without gap
    gap = "0d"
    gap_num = 0

    def max_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, max, min_periods=min_periods)

    rolling_max_obj = roll_series_with_gap(
        high_frequency_series,
        window_length,
        min_periods=min_periods,
        gap=gap,
    )
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    assert rolling_max_series.isna().sum() == (min_periods - 1) + gap_num

    # Check with small gap
    gap = "3H"
    gap_num = 3

    def max_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, max, min_periods=min_periods)

    rolling_max_obj = roll_series_with_gap(
        high_frequency_series,
        window_length,
        min_periods=min_periods,
        gap=gap,
    )
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    assert rolling_max_series.isna().sum() == (min_periods - 1) + gap_num

    # Check with large gap - in terms of days, so we'll multiply by 24hours for number of nans
    gap = "2D"
    gap_num = 2

    def max_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, max, min_periods=min_periods)

    rolling_max_obj = roll_series_with_gap(
        high_frequency_series,
        window_length,
        min_periods=min_periods,
        gap=gap,
    )
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    assert rolling_max_series.isna().sum() == (min_periods - 1) + (gap_num * 24)


def test_apply_roll_with_offset_data_min_periods_too_big(window_series_pd):
    window_length = "5D"
    gap = "2d"

    # Since the data has a daily frequency, there will only be, at most, 5 rows in the window
    min_periods = 6

    def max_wrapper(sub_s):
        return apply_roll_with_offset_gap(sub_s, gap, max, min_periods=min_periods)

    rolling_max_obj = roll_series_with_gap(
        window_series_pd,
        window_length,
        min_periods=min_periods,
        gap=gap,
    )
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    # The resulting series is comprised entirely of nans
    assert rolling_max_series.isna().sum() == len(window_series_pd)


def test_roll_series_with_gap_different_input_types_same_result_uniform(
    window_series_pd,
):
    # Offset inputs will only produce the same results as numeric inputs
    # when the data has a uniform frequency
    offset_gap = "2d"
    offset_window_length = "5d"
    int_gap = 2
    int_window_length = 5
    min_periods = 1

    # Rolling series' with matching input types
    expected_rolling_numeric = roll_series_with_gap(
        window_series_pd,
        window_length=int_window_length,
        gap=int_gap,
        min_periods=min_periods,
    ).max()

    def count_wrapper(sub_s):
        return apply_roll_with_offset_gap(
            sub_s,
            offset_gap,
            max,
            min_periods=min_periods,
        )

    rolling_count_obj = roll_series_with_gap(
        window_series_pd,
        window_length=offset_window_length,
        gap=offset_gap,
        min_periods=min_periods,
    )
    expected_rolling_offset = rolling_count_obj.apply(count_wrapper)

    # confirm that the offset and gap results are equal to one another
    pd.testing.assert_series_equal(expected_rolling_numeric, expected_rolling_offset)

    # Rolling series' with mismatched input types
    mismatched_numeric_gap = roll_series_with_gap(
        window_series_pd,
        window_length=offset_window_length,
        gap=int_gap,
        min_periods=min_periods,
    ).max()
    # Confirm the mismatched results also produce the same results
    pd.testing.assert_series_equal(expected_rolling_numeric, mismatched_numeric_gap)


def test_roll_series_with_gap_incorrect_types(window_series_pd):
    error = "Window length must be either an offset string or an integer."
    with pytest.raises(TypeError, match=error):
        (
            roll_series_with_gap(
                window_series_pd,
                window_length=4.2,
                gap=4,
                min_periods=1,
            ),
        )

    error = "Gap must be either an offset string or an integer."
    with pytest.raises(TypeError, match=error):
        roll_series_with_gap(window_series_pd, window_length=4, gap=4.2, min_periods=1)


def test_roll_series_with_gap_negative_inputs(window_series_pd):
    error = "Window length must be greater than zero."
    with pytest.raises(ValueError, match=error):
        roll_series_with_gap(window_series_pd, window_length=-4, gap=4, min_periods=1)

    error = "Gap must be greater than or equal to zero."
    with pytest.raises(ValueError, match=error):
        roll_series_with_gap(window_series_pd, window_length=4, gap=-4, min_periods=1)


def test_roll_series_with_non_offset_string_inputs(window_series_pd):
    error = "Cannot roll series. The specified gap, test, is not a valid offset alias."
    with pytest.raises(ValueError, match=error):
        roll_series_with_gap(
            window_series_pd,
            window_length="4D",
            gap="test",
            min_periods=1,
        )

    error = "Cannot roll series. The specified window length, test, is not a valid offset alias."
    with pytest.raises(ValueError, match=error):
        roll_series_with_gap(
            window_series_pd,
            window_length="test",
            gap="7D",
            min_periods=1,
        )

    # Test mismatched types error
    error = (
        "Cannot roll series with offset gap, 2d, and numeric window length, 7. "
        "If an offset alias is used for gap, the window length must also be defined as an offset alias. "
        "Please either change gap to be numeric or change window length to be an offset alias."
    )
    with pytest.raises(TypeError, match=error):
        roll_series_with_gap(
            window_series_pd,
            window_length=7,
            gap="2d",
            min_periods=1,
        ).max()


@pytest.mark.parametrize(
    "primitive",
    [RollingCount, RollingMax, RollingMin, RollingMean, RollingSTD, RollingTrend],
)
@patch(
    "featuretools.primitives.standard.transform.time_series.utils.apply_roll_with_offset_gap",
)
def test_no_call_to_apply_roll_with_offset_gap_with_numeric(
    mock_apply_roll,
    primitive,
    window_series_pd,
):
    assert not mock_apply_roll.called

    fully_numeric_primitive = primitive(window_length=3, gap=1)
    primitive_func = fully_numeric_primitive.get_function()
    if isinstance(fully_numeric_primitive, RollingCount):
        pd.Series(primitive_func(window_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                window_series_pd.index,
                pd.Series(window_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    offset_window_primitive = primitive(window_length="3d", gap=1)
    primitive_func = offset_window_primitive.get_function()
    if isinstance(offset_window_primitive, RollingCount):
        pd.Series(primitive_func(window_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                window_series_pd.index,
                pd.Series(window_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    no_gap_specified_primitive = primitive(window_length="3d")
    primitive_func = no_gap_specified_primitive.get_function()
    if isinstance(no_gap_specified_primitive, RollingCount):
        pd.Series(primitive_func(window_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                window_series_pd.index,
                pd.Series(window_series_pd.values),
            ),
        )

    assert not mock_apply_roll.called

    no_gap_specified_primitive = primitive(window_length="3d", gap="1d")
    primitive_func = no_gap_specified_primitive.get_function()
    if isinstance(no_gap_specified_primitive, RollingCount):
        pd.Series(primitive_func(window_series_pd.index))
    else:
        pd.Series(
            primitive_func(
                window_series_pd.index,
                pd.Series(window_series_pd.values),
            ),
        )

    assert mock_apply_roll.called
