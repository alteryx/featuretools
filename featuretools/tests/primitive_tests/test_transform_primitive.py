from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import Age, TimeSince, Week


def test_time_since():
    time_since = TimeSince()
    # class datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,
    times = pd.Series([datetime(2019, 3, 1, 0, 0, 0, 1),
                       datetime(2019, 3, 1, 0, 0, 1, 0),
                       datetime(2019, 3, 1, 0, 2, 0, 0)])
    cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
    values = time_since(array=times, time=cutoff_time)

    assert(list(map(int, values)) == [0, -1, -120])

    time_since = TimeSince(unit='nanoseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(round, values)) == [-1000, -1000000000, -120000000000])

    time_since = TimeSince(unit='milliseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1000, -120000])

    time_since = TimeSince(unit='Milliseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1000, -120000])

    time_since = TimeSince(unit='Years')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, 0, 0])

    times_y = pd.Series([datetime(2019, 3, 1, 0, 0, 0, 1),
                         datetime(2020, 3, 1, 0, 0, 1, 0),
                         datetime(2017, 3, 1, 0, 0, 0, 0)])

    time_since = TimeSince(unit='Years')
    values = time_since(array=times_y, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1, 1])

    error_text = 'Invalid unit given, make sure it is plural'
    with pytest.raises(ValueError, match=error_text):
        time_since = TimeSince(unit='na')
        time_since(array=times, time=cutoff_time)


def test_age():
    age = Age()
    dates = pd.Series(datetime(2010, 2, 26))
    ages = age(dates, time=datetime(2020, 2, 26))
    correct_ages = [10.005]   # .005 added due to leap years
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)


def test_age_two_years_quarterly():
    age = Age()
    dates = pd.Series(pd.date_range('2010-01-01', '2011-12-31', freq='Q'))
    ages = age(dates, time=datetime(2020, 2, 26))
    correct_ages = [9.915, 9.666, 9.414, 9.162, 8.915, 8.666, 8.414, 8.162]
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)


def test_age_leap_year():
    age = Age()
    dates = pd.Series([datetime(2016, 1, 1)])
    ages = age(dates, time=datetime(2016, 3, 1))
    correct_ages = [(31 + 29) / 365.0]
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)
    # born leap year date
    dates = pd.Series([datetime(2016, 2, 29)])
    ages = age(dates, time=datetime(2020, 2, 29))
    correct_ages = [4.0027]  # .0027 added due to leap year
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)


def test_age_nan():
    age = Age()
    dates = pd.Series([datetime(2010, 1, 1), np.nan, datetime(2012, 1, 1)])
    ages = age(dates, time=datetime(2020, 2, 26))
    correct_ages = [10.159, np.nan, 8.159]
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)


def test_week_no_deprecation_message():
    dates = [datetime(2019, 1, 3),
             datetime(2019, 6, 17, 11, 10, 50),
             datetime(2019, 11, 30, 19, 45, 15)
             ]
    with pytest.warns(None) as record:
        week = Week()
        week(dates).tolist()
    assert not record
