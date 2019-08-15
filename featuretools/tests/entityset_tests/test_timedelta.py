import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta
from toolz import merge

import featuretools as ft
from featuretools.entityset import Timedelta
from featuretools.primitives import Count  # , SlidingMean
from featuretools.utils.wrangle import _check_timedelta


def test_timedelta_equality():
    assert Timedelta(10, "d") == Timedelta(10, "d")
    assert Timedelta(10, "d") != 1


def test_singular():
    assert Timedelta.make_singular("Month") == "Month"
    assert Timedelta.make_singular("Months") == "Month"


def test_delta_with_observations(es):
    four_delta = Timedelta(4, 'observations')
    assert not four_delta.is_absolute()
    assert four_delta.value == 4

    neg_four_delta = -four_delta
    assert not neg_four_delta.is_absolute()
    assert neg_four_delta.value == -4

    time = pd.to_datetime('2019-05-01')

    error_txt = 'Invalid unit'
    with pytest.raises(Exception, match=error_txt):
        time + four_delta

    with pytest.raises(Exception, match=error_txt):
        time - four_delta


def test_delta_with_time_unit_matches_pandas(es):
    customer_id = 0
    sessions_df = es['sessions'].df
    sessions_df = sessions_df[sessions_df['customer_id'] == customer_id]
    log_df = es['log'].df
    log_df = log_df[log_df['session_id'].isin(sessions_df['id'])]
    all_times = log_df['datetime'].sort_values().tolist()

    # 4 observation delta
    value = 4
    unit = 'h'
    delta = Timedelta(value, unit)
    neg_delta = -delta
    # first plus 4 obs is fifth
    assert all_times[0] + delta == all_times[0] + pd.Timedelta(value, unit)
    # using negative
    assert all_times[0] - neg_delta == all_times[0] + pd.Timedelta(value, unit)

    # fifth minus 4 obs is first
    assert all_times[4] - delta == all_times[4] - pd.Timedelta(value, unit)
    # using negative
    assert all_times[4] + neg_delta == all_times[4] - pd.Timedelta(value, unit)


def test_check_timedelta(es):
    time_units = list(Timedelta._readable_units.keys())
    expanded_units = list(Timedelta._readable_units.values())
    exp_to_standard_unit = {e: t for e, t in zip(expanded_units, time_units)}
    singular_units = [u[:-1] for u in expanded_units]
    sing_to_standard_unit = {s: t for s, t in zip(singular_units, time_units)}
    to_standard_unit = merge(exp_to_standard_unit, sing_to_standard_unit)
    full_units = singular_units + expanded_units + time_units + time_units

    strings = ["2 {}".format(u) for u in singular_units + expanded_units +
               time_units]
    strings += ["2{}".format(u) for u in time_units]
    for i, s in enumerate(strings):
        unit = full_units[i]
        standard_unit = unit
        if unit in to_standard_unit:
            standard_unit = to_standard_unit[unit]

        td = _check_timedelta(s)
        if standard_unit != 'w':
            assert td.value == 2
            assert td.unit == standard_unit
        else:
            assert td.value == 2 * 7


def test_check_pd_timedelta(es):
    pdtd = pd.Timedelta(5, 'm')
    td = _check_timedelta(pdtd)
    assert td.unit == 's'
    assert td.value == 300


def test_week_to_days():
    assert Timedelta("1001 weeks") == Timedelta(1001 * 7, "days")


def test_string_timedelta_args():
    assert Timedelta("1 second") == Timedelta(1, "second")
    assert Timedelta("1 seconds") == Timedelta(1, "second")
    assert Timedelta("10 days") == Timedelta(10, "days")
    assert Timedelta("100 days") == Timedelta(100, "days")
    assert Timedelta("1001 days") == Timedelta(1001, "days")
    assert Timedelta("1001 weeks") == Timedelta(1001, "weeks")


def test_feature_takes_timedelta_string(es):
    feature = ft.Feature(es['log']['id'], parent_entity=es['customers'],
                         use_previous="1 day", primitive=Count)
    assert feature.use_previous == Timedelta(1, 'd')


# def test_sliding_feature_takes_timedelta_string(es):
#     feature = SlidingMean(es['log']['id'], es['customers'],
#                           use_previous="1 day",
#                           window_size="1 second")
#     assert feature.use_previous == Timedelta(1, 'd')
#     assert feature.window_size == Timedelta(1, 's')


def test_deltas_week(es):
    customer_id = 0
    sessions_df = es['sessions'].df
    sessions_df = sessions_df[sessions_df['customer_id'] == customer_id]
    log_df = es['log'].df
    log_df = log_df[log_df['session_id'].isin(sessions_df['id'])]
    all_times = log_df['datetime'].sort_values().tolist()
    delta_week = Timedelta(1, "w")
    delta_days = Timedelta(7, "d")

    assert all_times[0] + delta_days == all_times[0] + delta_week


def test_relative_year():
    td_time = "1 years"
    td = _check_timedelta(td_time)
    assert td.unit == "Y"
    assert td.value == 1
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime('2020-02-29')
    assert time + td == pd.to_datetime('2021-02-28')


def test_serialization():
    times = [
        Timedelta(1, unit='w'),
        Timedelta(3, unit='d'),
        Timedelta(5, unit='o'),
    ]

    dictionaries = [
        {'value': 1, 'unit': 'w'},
        {'value': 3, 'unit': 'd'},
        {'value': 5, 'unit': 'o'}
    ]

    for td, expected in zip(times, dictionaries):
        assert expected == td.get_arguments()

    for expected, dictionary in zip(times, dictionaries):
        assert expected == Timedelta.from_dictionary(dictionary)


def test_relative_month():
    td_time = "1 month"
    td = _check_timedelta(td_time)
    assert td.unit == "mo"
    assert td.value == 1
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime('2020-01-31')
    assert time + td == pd.to_datetime('2020-02-29')

    td_time = "6 months"
    td = _check_timedelta(td_time)
    assert td.unit == "mo"
    assert td.value == 6
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime('2020-01-31')
    assert time + td == pd.to_datetime('2020-07-31')
