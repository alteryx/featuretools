import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from featuretools.entityset import Timedelta
from featuretools.feature_base import Feature
from featuretools.primitives import Count
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.wrangle import _check_timedelta


def test_timedelta_equality():
    assert Timedelta(10, "d") == Timedelta(10, "d")
    assert Timedelta(10, "d") != 1


def test_singular():
    assert Timedelta.make_singular("Month") == "Month"
    assert Timedelta.make_singular("Months") == "Month"


def test_delta_with_observations(es):
    four_delta = Timedelta(4, "observations")
    assert not four_delta.is_absolute()
    assert four_delta.get_value("o") == 4

    neg_four_delta = -four_delta
    assert not neg_four_delta.is_absolute()
    assert neg_four_delta.get_value("o") == -4

    time = pd.to_datetime("2019-05-01")

    error_txt = "Invalid unit"
    with pytest.raises(Exception, match=error_txt):
        time + four_delta

    with pytest.raises(Exception, match=error_txt):
        time - four_delta


def test_delta_with_time_unit_matches_pandas(es):
    customer_id = 0
    sessions_df = to_pandas(es["sessions"])
    sessions_df = sessions_df[sessions_df["customer_id"] == customer_id]
    log_df = to_pandas(es["log"])
    log_df = log_df[log_df["session_id"].isin(sessions_df["id"])]
    all_times = log_df["datetime"].sort_values().tolist()

    # 4 observation delta
    value = 4
    unit = "h"
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
    to_standard_unit = {}
    to_standard_unit.update(exp_to_standard_unit)
    to_standard_unit.update(sing_to_standard_unit)
    full_units = singular_units + expanded_units + time_units + time_units

    strings = ["2 {}".format(u) for u in singular_units + expanded_units + time_units]
    strings += ["2{}".format(u) for u in time_units]
    for i, s in enumerate(strings):
        unit = full_units[i]
        standard_unit = unit
        if unit in to_standard_unit:
            standard_unit = to_standard_unit[unit]

        td = _check_timedelta(s)
        assert td.get_value(standard_unit) == 2


def test_check_pd_timedelta(es):
    pdtd = pd.Timedelta(5, "m")
    td = _check_timedelta(pdtd)
    assert td.get_value("s") == 300


def test_string_timedelta_args():
    assert Timedelta("1 second") == Timedelta(1, "second")
    assert Timedelta("1 seconds") == Timedelta(1, "second")
    assert Timedelta("10 days") == Timedelta(10, "days")
    assert Timedelta("100 days") == Timedelta(100, "days")
    assert Timedelta("1001 days") == Timedelta(1001, "days")
    assert Timedelta("1001 weeks") == Timedelta(1001, "weeks")


def test_feature_takes_timedelta_string(es):
    feature = Feature(
        Feature(es["log"].ww["id"]),
        parent_dataframe_name="customers",
        use_previous="1 day",
        primitive=Count,
    )
    assert feature.use_previous == Timedelta(1, "d")


def test_deltas_week(es):
    customer_id = 0
    sessions_df = to_pandas(es["sessions"])
    sessions_df = sessions_df[sessions_df["customer_id"] == customer_id]
    log_df = to_pandas(es["log"])
    log_df = log_df[log_df["session_id"].isin(sessions_df["id"])]
    all_times = log_df["datetime"].sort_values().tolist()
    delta_week = Timedelta(1, "w")
    delta_days = Timedelta(7, "d")

    assert all_times[0] + delta_days == all_times[0] + delta_week


def test_relative_year():
    td_time = "1 years"
    td = _check_timedelta(td_time)
    assert td.get_value("Y") == 1
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime("2020-02-29")
    assert time + td == pd.to_datetime("2021-02-28")


def test_serialization():
    times = [Timedelta(1, unit="w"), Timedelta(3, unit="d"), Timedelta(5, unit="o")]

    dictionaries = [
        {"value": 1, "unit": "w"},
        {"value": 3, "unit": "d"},
        {"value": 5, "unit": "o"},
    ]

    for td, expected in zip(times, dictionaries):
        assert expected == td.get_arguments()

    for expected, dictionary in zip(times, dictionaries):
        assert expected == Timedelta.from_dictionary(dictionary)

    # Test multiple temporal parameters separately since it is not deterministic
    mult_time = {"years": 4, "months": 3, "days": 2}
    mult_td = Timedelta(mult_time)

    # Serialize
    td_units = mult_td.get_arguments()["unit"]
    td_values = mult_td.get_arguments()["value"]
    arg_list = list(zip(td_values, td_units))

    assert (4, "Y") in arg_list
    assert (3, "mo") in arg_list
    assert (2, "d") in arg_list

    # Deserialize
    assert mult_td == Timedelta.from_dictionary(
        {"value": [4, 3, 2], "unit": ["Y", "mo", "d"]},
    )


def test_relative_month():
    td_time = "1 month"
    td = _check_timedelta(td_time)
    assert td.get_value("mo") == 1
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime("2020-01-31")
    assert time + td == pd.to_datetime("2020-02-29")

    td_time = "6 months"
    td = _check_timedelta(td_time)
    assert td.get_value("mo") == 6
    assert isinstance(td.delta_obj, relativedelta)

    time = pd.to_datetime("2020-01-31")
    assert time + td == pd.to_datetime("2020-07-31")


def test_has_multiple_units():
    single_unit = pd.DateOffset(months=3)
    multiple_units = pd.DateOffset(months=3, years=3, days=5)
    single_td = _check_timedelta(single_unit)
    multiple_td = _check_timedelta(multiple_units)
    assert single_td.has_multiple_units() is False
    assert multiple_td.has_multiple_units() is True


def test_pd_dateoffset_to_timedelta():
    single_temporal = pd.DateOffset(months=3)
    single_td = _check_timedelta(single_temporal)
    assert single_td.get_value("mo") == 3
    assert single_td.delta_obj == pd.DateOffset(months=3)

    mult_temporal = pd.DateOffset(years=10, months=3, days=5)
    mult_td = _check_timedelta(mult_temporal)
    expected = {"Y": 10, "mo": 3, "d": 5}
    assert mult_td.get_value() == expected
    assert mult_td.delta_obj == mult_temporal
    # get_name() for multiple values is not deterministic
    assert len(mult_td.get_name()) == len("10 Years 3 Months 5 Days")

    special_dateoffset = pd.offsets.BDay(100)
    special_td = _check_timedelta(special_dateoffset)
    assert special_td.get_value("businessdays") == 100
    assert special_td.delta_obj == special_dateoffset


def test_pd_dateoffset_to_timedelta_math():
    base = pd.to_datetime("2020-01-31")
    add = _check_timedelta(pd.DateOffset(months=2))
    res = base + add
    assert res == pd.to_datetime("2020-03-31")

    base_2 = pd.to_datetime("2020-01-31")
    add_2 = _check_timedelta(pd.DateOffset(months=2, days=3))
    res_2 = base_2 + add_2
    assert res_2 == pd.to_datetime("2020-04-03")

    base_3 = pd.to_datetime("2019-09-20")
    sub = _check_timedelta(pd.offsets.BDay(10))
    res_3 = base_3 - sub
    assert res_3 == pd.to_datetime("2019-09-06")
