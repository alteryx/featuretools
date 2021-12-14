import os

import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import pytest

from featuretools import list_primitives
from featuretools.primitives import (
    Age,
    Count,
    Day,
    GreaterThan,
    Haversine,
    Last,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    NumCharacters,
    NumUnique,
    NumWords,
    PercentTrue,
    Skew,
    Std,
    Sum,
    Weekday,
    Year,
    get_aggregation_primitives,
    get_default_aggregation_primitives,
    get_default_transform_primitives,
    get_transform_primitives
)
from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.utils import (
    _apply_roll_with_offset_gap,
    _get_descriptions,
    _get_num_gap_rows_from_offset,
    _get_unique_input_types,
    _roll_series_with_gap,
    list_primitive_files,
    load_primitive_from_file
)
from featuretools.tests.primitive_tests.utils import get_number_of_days
from featuretools.utils.gen_utils import Library


def test_list_primitives_order():
    df = list_primitives()
    all_primitives = get_transform_primitives()
    all_primitives.update(get_aggregation_primitives())

    for name, primitive in all_primitives.items():
        assert name in df['name'].values
        row = df.loc[df['name'] == name].iloc[0]
        actual_desc = _get_descriptions([primitive])[0]
        if actual_desc:
            assert actual_desc == row['description']
        assert row['dask_compatible'] == (Library.DASK in primitive.compatibility)
        assert row['valid_inputs'] == ', '.join(_get_unique_input_types(primitive.input_types))
        assert row['return_type'] == getattr(primitive.return_type, '__name__', None)

    types = df['type'].values
    assert 'aggregation' in types
    assert 'transform' in types


def test_valid_input_types():
    actual = _get_unique_input_types(Haversine.input_types)
    assert actual == {'<ColumnSchema (Logical Type = LatLong)>'}
    actual = _get_unique_input_types(GreaterThan.input_types)
    assert actual == {'<ColumnSchema (Logical Type = Datetime)>',
                      "<ColumnSchema (Semantic Tags = ['numeric'])>",
                      '<ColumnSchema (Logical Type = Ordinal)>'}
    actual = _get_unique_input_types(Sum.input_types)
    assert actual == {"<ColumnSchema (Semantic Tags = ['numeric'])>"}


def test_descriptions():
    primitives = {NumCharacters: 'Calculates the number of characters in a string.',
                  Day: 'Determines the day of the month from a datetime.',
                  Last: 'Determines the last value in a list.',
                  GreaterThan: 'Determines if values in one list are greater than another list.'}
    assert _get_descriptions(list(primitives.keys())) == list(primitives.values())


def test_get_default_aggregation_primitives():
    primitives = get_default_aggregation_primitives()
    expected_primitives = [Sum, Std, Max, Skew, Min, Mean, Count, PercentTrue,
                           NumUnique, Mode]
    assert set(primitives) == set(expected_primitives)


def test_get_default_transform_primitives():
    primitives = get_default_transform_primitives()
    expected_primitives = [Age, Day, Year, Month, Weekday, Haversine, NumWords,
                           NumCharacters]
    assert set(primitives) == set(expected_primitives)


@pytest.fixture
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def primitives_to_install_dir(this_dir):
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture
def bad_primitives_files_dir(this_dir):
    return os.path.join(this_dir, "bad_primitive_files")


def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    custom_max_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    custom_mean_file = os.path.join(primitives_to_install_dir, "custom_mean.py")
    custom_sum_file = os.path.join(primitives_to_install_dir, "custom_sum.py")
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))


def test_load_primitive_from_file(primitives_to_install_dir):
    primitve_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    primitive_name, primitive_obj = load_primitive_from_file(primitve_file)
    assert issubclass(primitive_obj, PrimitiveBase)


def test_errors_more_than_one_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "multiple_primitives.py")
    error_text = "More than one primitive defined in file {}".format(primitive_file)
    with pytest.raises(RuntimeError) as excinfo:
        load_primitive_from_file(primitive_file)
    assert str(excinfo.value) == error_text


def test_errors_no_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "no_primitives.py")
    error_text = "No primitive defined in file {}".format(primitive_file)
    with pytest.raises(RuntimeError) as excinfo:
        load_primitive_from_file(primitive_file)
    assert str(excinfo.value) == error_text


def test_get_num_gap_rows_from_offset(rolling_series_pd):
    # Data is daily, so number of rows should be number of days

    assert _get_num_gap_rows_from_offset(rolling_series_pd, "10D") == 10
    assert _get_num_gap_rows_from_offset(rolling_series_pd, "0D") == 0
    assert _get_num_gap_rows_from_offset(rolling_series_pd, "48H") == 2
    assert _get_num_gap_rows_from_offset(rolling_series_pd, "4H") == 1


def test_get_num_gap_rows_from_offset_not_uniform(rolling_series_pd):
    non_uniform_series = rolling_series_pd.iloc[[0, 2, 5, 6, 8, 9]]

    assert _get_num_gap_rows_from_offset(non_uniform_series, "10D") == 6
    assert _get_num_gap_rows_from_offset(non_uniform_series, "0D") == 0
    assert _get_num_gap_rows_from_offset(non_uniform_series, "48H") == 1
    assert _get_num_gap_rows_from_offset(non_uniform_series, "4H") == 1
    assert _get_num_gap_rows_from_offset(non_uniform_series, "5D") == 2


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (3, 2),
        (3, 4),  # gap larger than window
        (2, 0),  # gap explicitly set to 0
        ('3d', '2d'),  # using offset aliases
        ('3d', '4d'),  # using offset aliases
        ('4d', '0d'),
    ],
)
def test_roll_series_with_gap(window_length, gap, rolling_series_pd):
    rolling_max = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap).max()
    rolling_min = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap).min()

    assert len(rolling_max) == len(rolling_series_pd)
    assert len(rolling_min) == len(rolling_series_pd)

    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)
    for i in range(len(rolling_series_pd)):
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


@pytest.mark.parametrize(
    "window_length",
    [
        3,
        "3d"
    ]
)
def test_roll_series_with_no_gap(window_length, rolling_series_pd):
    actual_rolling = _roll_series_with_gap(rolling_series_pd, window_length).mean()
    expected_rolling = rolling_series_pd.rolling(window_length, min_periods=1).mean()

    pd.testing.assert_series_equal(actual_rolling, expected_rolling)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (6, 2),
        (6, 0),  # No gap - changes early values
        ('6d', '0d'),  # Uses offset aliases
        ('6d', '2d')  # Uses offset aliases
    ]
)
def test_roll_series_with_gap_early_values(window_length, gap, rolling_series_pd):
    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)

    # Default min periods is 1 - will include all
    default_partial_values = _roll_series_with_gap(rolling_series_pd,
                                                   window_length,
                                                   gap=gap).count()
    num_empty_aggregates = len(default_partial_values.loc[default_partial_values == 0])
    num_partial_aggregates = len((default_partial_values
                                  .loc[default_partial_values != 0])
                                 .loc[default_partial_values < window_length_num])

    assert num_partial_aggregates == window_length_num - 1
    if isinstance(gap, str):
        # gap isn't handled, so we'll always at least include the row itself
        assert num_empty_aggregates == 0
    else:
        assert num_empty_aggregates == gap_num

    # Make min periods the size of the window
    no_partial_values = _roll_series_with_gap(rolling_series_pd,
                                              window_length,
                                              gap=gap,
                                              min_periods=window_length_num).count()
    num_null_aggregates = len(no_partial_values.loc[pd.isna(no_partial_values)])
    num_partial_aggregates = len(no_partial_values.loc[no_partial_values < window_length_num])

    # because we shift, gap is included as nan values in the series.
    # Count treats nans in a window as values that don't get counted,
    # so the gap rows get included in the count for whether a window has "min periods".
    # This is different than max, for example, which does not count nans in a window as values towards "min periods"
    assert num_null_aggregates == window_length_num - 1
    if isinstance(gap, str):
        # gap isn't handled, so we'll never have any partial aggregates that would have come in the gap'
        assert num_partial_aggregates == 0
    else:
        assert num_partial_aggregates == gap_num


def test_roll_series_with_gap_nullable_types(rolling_series_pd):
    window_length = 3
    gap = 2
    # Because we're inserting nans, confirm that nullability of the dtype doesn't have an impact on the results
    nullable_series = rolling_series_pd.astype('Int64')
    non_nullable_series = rolling_series_pd.astype('int64')

    nullable_rolling_max = _roll_series_with_gap(nullable_series, window_length, gap=gap).max()
    non_nullable_rolling_max = _roll_series_with_gap(non_nullable_series, window_length, gap=gap).max()

    pd.testing.assert_series_equal(nullable_rolling_max, non_nullable_rolling_max)


def test_roll_series_with_gap_nullable_types_with_nans(rolling_series_pd):
    window_length = 3
    gap = 2
    nullable_floats = rolling_series_pd.astype('float64').replace({1: np.nan, 3: np.nan})
    nullable_ints = nullable_floats.astype('Int64')

    nullable_ints_rolling_max = _roll_series_with_gap(nullable_ints, window_length, gap=gap).max()
    nullable_floats_rolling_max = _roll_series_with_gap(nullable_floats, window_length, gap=gap).max()

    pd.testing.assert_series_equal(nullable_ints_rolling_max, nullable_floats_rolling_max)

    expected_early_values = ([np.nan, np.nan, 0, 0, 2, 2, 4] +
                             list(range(7 - gap, len(rolling_series_pd) - gap)))
    for i in range(len(rolling_series_pd)):
        actual = nullable_floats_rolling_max.iloc[i]
        expected = expected_early_values[i]

        if pd.isnull(actual):
            assert pd.isnull(expected)
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "window_length, gap",
    [
        ('3d', '2d'),
        ('3d', '4d'),
        ('4d', '0d'),
    ],
)
def test_apply_roll_with_offset_gap(window_length, gap, rolling_series_pd):
    def max_wrapper(sub_s):
        return _apply_roll_with_offset_gap(sub_s, gap, max, min_periods=1)

    def min_wrapper(sub_s):
        return _apply_roll_with_offset_gap(sub_s, gap, min, min_periods=1)

    rolling_max_obj = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap)
    rolling_max_series = rolling_max_obj.apply(max_wrapper)

    rolling_min_obj = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap)
    rolling_min_series = rolling_min_obj.apply(min_wrapper)

    assert len(rolling_max_series) == len(rolling_series_pd)
    assert len(rolling_min_series) == len(rolling_series_pd)

    gap_num = get_number_of_days(gap)
    window_length_num = get_number_of_days(window_length)
    for i in range(len(rolling_series_pd)):
        start_idx = i - gap_num - window_length_num + 1
        # Now this acts as expected
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
def test_apply_roll_with_offset_gap_default_min_periods(min_periods, rolling_series_pd):
    window_length = '5d'
    window_length_num = 5
    gap = '3d'
    gap_num = 3

    def count_wrapper(sub_s):
        return _apply_roll_with_offset_gap(sub_s, gap, len, min_periods=min_periods)

    rolling_count_obj = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap)
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    # gap essentially creates rolling series that have no elements; which should be nan
    # to differentiate from when a window only has null values
    num_empty_aggregates = rolling_count_series.isna().sum()
    num_partial_aggregates = len((rolling_count_series
                                  .loc[rolling_count_series != 0])
                                 .loc[rolling_count_series < window_length_num])

    assert num_empty_aggregates == gap_num
    assert num_partial_aggregates == window_length_num - 1


@pytest.mark.parametrize(
    "min_periods",
    [2, 3, 4, 5],
)
def test_apply_roll_with_offset_gap_min_periods(min_periods, rolling_series_pd):
    window_length = '5d'
    window_length_num = 5
    gap = '3d'
    gap_num = 3

    def count_wrapper(sub_s):
        return _apply_roll_with_offset_gap(sub_s, gap, len, min_periods=min_periods)

    rolling_count_obj = _roll_series_with_gap(rolling_series_pd, window_length, gap=gap)
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    # gap essentially creates rolling series that have no elements; which should be nan
    # to differentiate from when a window only has null values
    num_empty_aggregates = rolling_count_series.isna().sum()
    num_partial_aggregates = len((rolling_count_series
                                  .loc[rolling_count_series != 0])
                                 .loc[rolling_count_series < window_length_num])

    assert num_empty_aggregates == min_periods - 1 + gap_num
    assert num_partial_aggregates == window_length_num - min_periods


def test_apply_roll_with_offset_gap_non_uniform():
    rows_to_change = [2, 6, 12]
    window_length = '3d'
    gap = '1d'

    # When the data isn't uniform, this impacts the number of values in each rolling window
    # pick rows that should be pushed backwards to be one day after the previous day
    # this means that with a window length of 7D, when we get to those rows, there's another value within the 7 days window
    # where there are none for the rest of the rows
    datetimes = list(pd.date_range(start='2017-01-01', freq='1W', periods=20))
    for i in rows_to_change:
        datetimes[i] = datetimes[i] - pd.Timedelta('6D')
    no_freq_series = pd.Series(range(20), index=datetimes)

    assert pd.infer_freq(no_freq_series.index) is None

    def count_wrapper(sub_s):
        return _apply_roll_with_offset_gap(sub_s, gap, len, min_periods=1)
    rolling_count_obj = _roll_series_with_gap(no_freq_series, window_length, gap=gap)
    rolling_count_series = rolling_count_obj.apply(count_wrapper)

    counts = rolling_count_series.value_counts()
    assert counts[1] == 3
    assert rolling_count_series.isna().sum() == 17

    for i in rows_to_change:
        assert rolling_count_series.iloc[i] == 1

# --> add test with min periods being huge bc the offset strings are way too big - min_p = 20, freq = hourly, window_size = 1d

# def test_roll_series_with_gap_invalid_offset_strings():
#     pass


# def test_roll_series_with_gap_no_datetime():
#     pass


# def test_roll_series_with_mismatched_parameters(rolling_series_pd):
#     # --> maybe no longer necessary to fail there
#     error = 'Cannot roll series when window_length, 4 is and gap is 2d; parameters are not the same type.'
#     with pytest.raises(TypeError, match=error):
#         _roll_series_with_gap(rolling_series_pd, 4, gap="2d")

#     error = 'Cannot roll series when window_length is 4d, and gap is 2; parameters are not the same type.'
#     with pytest.raises(TypeError, match=error):
#         _roll_series_with_gap(rolling_series_pd, "4d", gap=2)
#     # --> check when no gap passed in


# @pytest.mark.parametrize(
#     "window_length",
#     [
#         4,
#         "4d"
#     ]
# )
# def test_roll_series_with_no_gap_parameter_set(window_length, rolling_series_pd):
#     # --> confirm both are the same
#     _roll_series_with_gap(rolling_series_pd, window_length)

# --> will fail till updated - may no longer be possible to use non fixed so maye remoce
# def test_roll_series_with_gap_and_non_fixed_offset_gap():
#     # --> clean up more
#     # The offset of 1W is <Week: weekday=6>, so the gap will make the start
#     # the first 6th day of the week after the first datetime, not 7 days after the first datetime
#     starts_on_sunday = pd.Series(range(40), index=pd.date_range(start='2017-1-15', freq='1D', periods=40))

#     rolled_series = _roll_series_with_gap(starts_on_sunday, "5D", gap="1M", min_periods=1).max()
#     assert rolled_series.isna().sum() == 16

#     rolled_series = _roll_series_with_gap(starts_on_sunday, "5D", gap="1W", min_periods=1).max()
#     assert rolled_series.isna().sum() == 7

#     starts_on_monday = pd.Series(range(40), index=pd.date_range(start='2017-01-16', freq='1D', periods=40))

#     rolled_series = _roll_series_with_gap(starts_on_monday, "5D", gap="1W", min_periods=1).max()
#     assert rolled_series.isna().sum() == 6

#     rolled_series = _roll_series_with_gap(starts_on_monday, "5D", gap="1M", min_periods=1).max()
#     assert rolled_series.isna().sum() == 15
