from datetime import datetime, timedelta
from itertools import chain

import numpy as np
import pandas as pd
import pytest

from featuretools.utils import convert_time_units, make_temporal_cutoffs
from featuretools.utils.time_utils import (
    calculate_trend,
    convert_datetime_to_floats,
    convert_timedelta_to_floats,
)


def test_make_temporal_cutoffs():
    instance_ids = pd.Series(range(10))
    cutoffs = pd.date_range(start="1/2/2015", periods=10, freq="1d")
    temporal_cutoffs_by_nwindows = make_temporal_cutoffs(
        instance_ids,
        cutoffs,
        window_size="1h",
        num_windows=2,
    )

    assert temporal_cutoffs_by_nwindows.shape[0] == 20
    actual_instances = chain.from_iterable([[i, i] for i in range(10)])
    actual_times = [
        "1/1/2015 23:00:00",
        "1/2/2015 00:00:00",
        "1/2/2015 23:00:00",
        "1/3/2015 00:00:00",
        "1/3/2015 23:00:00",
        "1/4/2015 00:00:00",
        "1/4/2015 23:00:00",
        "1/5/2015 00:00:00",
        "1/5/2015 23:00:00",
        "1/6/2015 00:00:00",
        "1/6/2015 23:00:00",
        "1/7/2015 00:00:00",
        "1/7/2015 23:00:00",
        "1/8/2015 00:00:00",
        "1/8/2015 23:00:00",
        "1/9/2015 00:00:00",
        "1/9/2015 23:00:00",
        "1/10/2015 00:00:00",
        "1/10/2015 23:00:00",
        "1/11/2015 00:00:00",
        "1/11/2015 23:00:00",
    ]
    actual_times = [pd.Timestamp(c) for c in actual_times]

    for computed, actual in zip(
        temporal_cutoffs_by_nwindows["instance_id"],
        actual_instances,
    ):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_nwindows["time"], actual_times):
        assert computed == actual

    cutoffs = [pd.Timestamp("1/2/2015")] * 9 + [pd.Timestamp("1/3/2015")]
    starts = [pd.Timestamp("1/1/2015")] * 9 + [pd.Timestamp("1/2/2015")]
    actual_times = ["1/1/2015 00:00:00", "1/2/2015 00:00:00"] * 9
    actual_times += ["1/2/2015 00:00:00", "1/3/2015 00:00:00"]
    actual_times = [pd.Timestamp(c) for c in actual_times]
    temporal_cutoffs_by_wsz_start = make_temporal_cutoffs(
        instance_ids,
        cutoffs,
        window_size="1d",
        start=starts,
    )

    for computed, actual in zip(
        temporal_cutoffs_by_wsz_start["instance_id"],
        actual_instances,
    ):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_wsz_start["time"], actual_times):
        assert computed == actual

    cutoffs = [pd.Timestamp("1/2/2015")] * 9 + [pd.Timestamp("1/3/2015")]
    starts = [pd.Timestamp("1/1/2015")] * 10
    actual_times = ["1/1/2015 00:00:00", "1/2/2015 00:00:00"] * 9
    actual_times += ["1/1/2015 00:00:00", "1/3/2015 00:00:00"]
    actual_times = [pd.Timestamp(c) for c in actual_times]
    temporal_cutoffs_by_nw_start = make_temporal_cutoffs(
        instance_ids,
        cutoffs,
        num_windows=2,
        start=starts,
    )

    for computed, actual in zip(
        temporal_cutoffs_by_nw_start["instance_id"],
        actual_instances,
    ):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_nw_start["time"], actual_times):
        assert computed == actual


def test_convert_time_units():
    units = {
        "years": 31540000,
        "months": 2628000,
        "days": 86400,
        "hours": 3600,
        "minutes": 60,
        "seconds": 1,
        "milliseconds": 0.001,
        "nanoseconds": 0.000000001,
    }
    for each in units:
        assert convert_time_units(units[each] * 2, each) == 2
        assert np.isclose(convert_time_units(float(units[each] * 2), each), 2)

    error_text = "Invalid unit given, make sure it is plural"
    with pytest.raises(ValueError, match=error_text):
        convert_time_units("jnkwjgn", 10)


@pytest.mark.parametrize(
    "dt, expected_floats",
    [
        (
            pd.Series(
                [
                    datetime(2010, 1, 1, 11, 45, 0),
                    datetime(2010, 1, 1, 12, 55, 15),
                    datetime(2010, 1, 1, 11, 57, 30),
                    datetime(2010, 1, 1, 11, 12),
                    datetime(2010, 1, 1, 11, 12, 15),
                ],
            ),
            pd.Series([21039105.0, 21039175.25, 21039117.5, 21039072.0, 21039072.25]),
        ),
        (
            pd.Series(
                list(pd.date_range(start="2017-01-01", freq="1d", periods=3))
                + list(pd.date_range(start="2017-01-10", freq="2d", periods=4))
                + list(pd.date_range(start="2017-01-22", freq="1d", periods=7)),
            ),
            pd.Series(
                [
                    17167.0,
                    17168.0,
                    17169.0,
                    17176.0,
                    17178.0,
                    17180.0,
                    17182.0,
                    17188.0,
                    17189.0,
                    17190.0,
                    17191.0,
                    17192.0,
                    17193.0,
                    17194.0,
                ],
            ),
        ),
    ],
)
def test_convert_datetime_floats(dt, expected_floats):
    actual_floats = convert_datetime_to_floats(dt)
    pd.testing.assert_series_equal(pd.Series(actual_floats), expected_floats)


@pytest.mark.parametrize(
    "td, expected_floats",
    [
        (
            pd.Series(
                [
                    pd.Timedelta(2, "day"),
                    pd.Timedelta(120000000),
                    pd.Timedelta(48, "sec"),
                    pd.Timedelta(30, "min"),
                    pd.Timedelta(12, "hour"),
                ],
            ),
            pd.Series(
                [
                    2.0,
                    1.388888888888889e-06,
                    0.0005555555555555556,
                    0.020833333333333332,
                    0.5,
                ],
            ),
        ),
        (
            pd.Series(
                [
                    timedelta(days=4),
                    timedelta(milliseconds=4000000),
                    timedelta(hours=2, seconds=49),
                ],
            ),
            pd.Series([4.0, 0.0462962962962963, 0.08390046296296297]),
        ),
    ],
)
def test_convert_timedelta_to_floats(td, expected_floats):
    actual_floats = convert_timedelta_to_floats(td)
    pd.testing.assert_series_equal(pd.Series(actual_floats), expected_floats)


@pytest.mark.parametrize(
    "series,expected_trends",
    [
        (
            # using datetimes
            pd.Series(
                data=[0, 5, 10],
                index=[
                    datetime(2019, 1, 1),
                    datetime(2019, 1, 2),
                    datetime(2019, 1, 3),
                ],
            ),
            5.0,
        ),
        (
            # using pd.Timestamp
            pd.Series(
                data=[0, -5, 3],
                index=pd.date_range(start="2019-01-01", freq="1D", periods=3),
            ),
            1.4999999999999998,
        ),
        (
            pd.Series(
                data=[1, 2, 4, 8, 16],
                index=pd.date_range(start="2019-01-01", freq="1D", periods=5),
            ),
            3.6000000000000005,
        ),
        (
            # using pd.Timedelta with no change in time
            pd.Series(
                data=[1, 2, 3],
                index=[
                    pd.Timedelta(120000000),
                    pd.Timedelta(120000000),
                    pd.Timedelta(120000000),
                ],
            ),
            0,
        ),
    ],
)
def test_calculate_trend(series, expected_trends):
    actual_trends = calculate_trend(series)
    assert np.isclose(actual_trends, expected_trends)
