from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pytz import timezone

from featuretools.primitives import (
    Age,
    DateToTimeZone,
    DayOfYear,
    DaysInMonth,
    EmailAddressToDomain,
    FileExtension,
    IsFirstWeekOfMonth,
    IsFreeEmailDomain,
    IsLeapYear,
    IsLunchTime,
    IsMonthEnd,
    IsMonthStart,
    IsQuarterEnd,
    IsQuarterStart,
    IsWorkingHours,
    IsYearEnd,
    IsYearStart,
    Lag,
    NthWeekOfMonth,
    NumericLag,
    PartOfDay,
    Quarter,
    RateOfChange,
    TimeSince,
    URLToDomain,
    URLToProtocol,
    URLToTLD,
    Week,
    get_transform_primitives,
)
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


def test_time_since():
    time_since = TimeSince()
    # class datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,
    times = pd.Series(
        [
            datetime(2019, 3, 1, 0, 0, 0, 1),
            datetime(2019, 3, 1, 0, 0, 1, 0),
            datetime(2019, 3, 1, 0, 2, 0, 0),
        ],
    )
    cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
    values = time_since(array=times, time=cutoff_time)

    assert list(map(int, values)) == [0, -1, -120]

    time_since = TimeSince(unit="nanoseconds")
    values = time_since(array=times, time=cutoff_time)
    assert list(map(round, values)) == [-1000, -1000000000, -120000000000]

    time_since = TimeSince(unit="milliseconds")
    values = time_since(array=times, time=cutoff_time)
    assert list(map(int, values)) == [0, -1000, -120000]

    time_since = TimeSince(unit="Milliseconds")
    values = time_since(array=times, time=cutoff_time)
    assert list(map(int, values)) == [0, -1000, -120000]

    time_since = TimeSince(unit="Years")
    values = time_since(array=times, time=cutoff_time)
    assert list(map(int, values)) == [0, 0, 0]

    times_y = pd.Series(
        [
            datetime(2019, 3, 1, 0, 0, 0, 1),
            datetime(2020, 3, 1, 0, 0, 1, 0),
            datetime(2017, 3, 1, 0, 0, 0, 0),
        ],
    )

    time_since = TimeSince(unit="Years")
    values = time_since(array=times_y, time=cutoff_time)
    assert list(map(int, values)) == [0, -1, 1]

    error_text = "Invalid unit given, make sure it is plural"
    with pytest.raises(ValueError, match=error_text):
        time_since = TimeSince(unit="na")
        time_since(array=times, time=cutoff_time)


def test_age():
    age = Age()
    dates = pd.Series(datetime(2010, 2, 26))
    ages = age(dates, time=datetime(2020, 2, 26))
    correct_ages = [10.005]  # .005 added due to leap years
    np.testing.assert_array_almost_equal(ages, correct_ages, decimal=3)


def test_age_two_years_quarterly():
    age = Age()
    dates = pd.Series(pd.date_range("2010-01-01", "2011-12-31", freq="Q"))
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


def test_day_of_year():
    doy = DayOfYear()
    dates = pd.Series([datetime(2019, 12, 31), np.nan, datetime(2020, 12, 31)])
    days_of_year = doy(dates)
    correct_days = [365, np.nan, 366]
    np.testing.assert_array_equal(days_of_year, correct_days)


def test_days_in_month():
    dim = DaysInMonth()
    dates = pd.Series(
        [datetime(2010, 1, 1), datetime(2019, 2, 1), np.nan, datetime(2020, 2, 1)],
    )
    days_in_month = dim(dates)
    correct_days = [31, 28, np.nan, 29]
    np.testing.assert_array_equal(days_in_month, correct_days)


def test_is_leap_year():
    ily = IsLeapYear()
    dates = pd.Series([datetime(2020, 1, 1), datetime(2021, 1, 1)])
    leap_year_bools = ily(dates)
    correct_bools = [True, False]
    np.testing.assert_array_equal(leap_year_bools, correct_bools)


def test_is_month_end():
    ime = IsMonthEnd()
    dates = pd.Series(
        [datetime(2019, 3, 1), datetime(2021, 2, 28), datetime(2020, 2, 29)],
    )
    ime_bools = ime(dates)
    correct_bools = [False, True, True]
    np.testing.assert_array_equal(ime_bools, correct_bools)


def test_is_month_start():
    ims = IsMonthStart()
    dates = pd.Series(
        [datetime(2019, 3, 1), datetime(2020, 2, 28), datetime(2020, 2, 29)],
    )
    ims_bools = ims(dates)
    correct_bools = [True, False, False]
    np.testing.assert_array_equal(ims_bools, correct_bools)


def test_is_quarter_end():
    iqe = IsQuarterEnd()
    dates = pd.Series([datetime(2020, 1, 1), datetime(2021, 3, 31)])
    iqe_bools = iqe(dates)
    correct_bools = [False, True]
    np.testing.assert_array_equal(iqe_bools, correct_bools)


def test_is_quarter_start():
    iqs = IsQuarterStart()
    dates = pd.Series([datetime(2020, 1, 1), datetime(2021, 3, 31)])
    iqs_bools = iqs(dates)
    correct_bools = [True, False]
    np.testing.assert_array_equal(iqs_bools, correct_bools)


def test_is_lunch_time_default():
    is_lunch_time = IsLunchTime()
    dates = pd.Series(
        [
            datetime(2022, 6, 26, 12, 12, 12),
            datetime(2022, 6, 28, 12, 3, 4),
            datetime(2022, 6, 28, 11, 3, 4),
            np.nan,
        ],
    )
    actual = is_lunch_time(dates)
    expected = [True, True, False, False]
    np.testing.assert_array_equal(actual, expected)


def test_is_lunch_time_configurable():
    is_lunch_time = IsLunchTime(14)
    dates = pd.Series(
        [
            datetime(2022, 6, 26, 12, 12, 12),
            datetime(2022, 6, 28, 14, 3, 4),
            datetime(2022, 6, 28, 11, 3, 4),
            np.nan,
        ],
    )
    actual = is_lunch_time(dates)
    expected = [False, True, False, False]
    np.testing.assert_array_equal(actual, expected)


def test_is_working_hours_standard_hours():
    is_working_hours = IsWorkingHours()
    dates = pd.Series(
        [
            datetime(2022, 6, 21, 16, 3, 3),
            datetime(2019, 1, 3, 4, 4, 4),
            datetime(2022, 1, 1, 12, 1, 2),
        ],
    )
    actual = is_working_hours(dates).tolist()
    expected = [True, False, True]
    np.testing.assert_array_equal(actual, expected)


def test_is_working_hours_configured_hours():
    is_working_hours = IsWorkingHours(15, 18)
    dates = pd.Series(
        [
            datetime(2022, 6, 21, 16, 3, 3),
            datetime(2022, 6, 26, 14, 4, 4),
            datetime(2022, 1, 1, 12, 1, 2),
        ],
    )
    answer = is_working_hours(dates).tolist()
    expected = [True, False, False]
    np.testing.assert_array_equal(answer, expected)


def test_part_of_day():
    pod = PartOfDay()
    dates = pd.Series(
        [
            datetime(2020, 1, 11, 0, 2, 1),
            datetime(2020, 1, 11, 1, 2, 1),
            datetime(2021, 3, 31, 4, 2, 1),
            datetime(2020, 3, 4, 6, 2, 1),
            datetime(2020, 3, 4, 8, 2, 1),
            datetime(2020, 3, 4, 11, 2, 1),
            datetime(2020, 3, 4, 14, 2, 3),
            datetime(2020, 3, 4, 17, 2, 3),
            datetime(2020, 2, 2, 20, 2, 2),
            np.nan,
        ],
    )
    actual = pod(dates)
    expected = pd.Series(
        [
            "midnight",
            "midnight",
            "dawn",
            "early morning",
            "late morning",
            "noon",
            "afternoon",
            "evening",
            "night",
            np.nan,
        ],
    )
    pd.testing.assert_series_equal(expected, actual)


def test_is_year_end():
    is_year_end = IsYearEnd()
    dates = pd.Series([datetime(2020, 12, 31), np.nan, datetime(2020, 1, 1)])
    answer = is_year_end(dates)
    correct_answer = [True, False, False]
    np.testing.assert_array_equal(answer, correct_answer)


def test_is_year_start():
    is_year_start = IsYearStart()
    dates = pd.Series([datetime(2020, 12, 31), np.nan, datetime(2020, 1, 1)])
    answer = is_year_start(dates)
    correct_answer = [False, False, True]
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter_regular():
    q = Quarter()
    array = pd.Series(
        [
            pd.to_datetime("2018-01-01"),
            pd.to_datetime("2018-04-01"),
            pd.to_datetime("2018-07-01"),
            pd.to_datetime("2018-10-01"),
        ],
    )
    answer = q(array)
    correct_answer = pd.Series([1, 2, 3, 4])
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter_leap_year():
    q = Quarter()
    array = pd.Series(
        [
            pd.to_datetime("2016-02-29"),
            pd.to_datetime("2018-04-01"),
            pd.to_datetime("2018-07-01"),
            pd.to_datetime("2018-10-01"),
        ],
    )
    answer = q(array)
    correct_answer = pd.Series([1, 2, 3, 4])
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter_nan_and_nat_input():
    q = Quarter()
    array = pd.Series(
        [
            pd.to_datetime("2016-02-29"),
            np.nan,
            np.datetime64("NaT"),
            pd.to_datetime("2018-10-01"),
        ],
    )
    answer = q(array)
    correct_answer = pd.Series([1, np.nan, np.nan, 4])
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter_year_before_1970():
    q = Quarter()
    array = pd.Series(
        [
            pd.to_datetime("2018-01-01"),
            pd.to_datetime("1950-04-01"),
            pd.to_datetime("1874-07-01"),
            pd.to_datetime("2018-10-01"),
        ],
    )
    answer = q(array)
    correct_answer = pd.Series([1, 2, 3, 4])
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter_year_after_2038():
    q = Quarter()
    array = pd.Series(
        [
            pd.to_datetime("2018-01-01"),
            pd.to_datetime("2050-04-01"),
            pd.to_datetime("2174-07-01"),
            pd.to_datetime("2018-10-01"),
        ],
    )
    answer = q(array)
    correct_answer = pd.Series([1, 2, 3, 4])
    np.testing.assert_array_equal(answer, correct_answer)


def test_quarter():
    q = Quarter()
    dates = [datetime(2019, 12, 1), datetime(2019, 1, 3), datetime(2020, 2, 1)]
    quarter = q(dates)
    correct_quarters = [4, 1, 1]
    np.testing.assert_array_equal(quarter, correct_quarters)


def test_week_no_deprecation_message():
    dates = [
        datetime(2019, 1, 3),
        datetime(2019, 6, 17, 11, 10, 50),
        datetime(2019, 11, 30, 19, 45, 15),
    ]
    with pytest.warns(None) as record:
        week = Week()
        week(dates).tolist()
    assert not record


def test_url_to_domain_urls():
    url_to_domain = URLToDomain()
    urls = pd.Series(
        [
            "https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22",
            "http://mplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://lplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://play.google.co.in/sadfask/asdkfals?dk=10",
            "http://tplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://www.google.co.in/sadfask/asdkfals?dk=10",
            "www.google.co.in/sadfask/asdkfals?dk=10",
            "http://user:pass@google.com/?a=b#asdd",
            "https://www.compzets.com?asd=10",
            "www.compzets.com?asd=10",
            "facebook.com",
            "https://www.compzets.net?asd=10",
            "http://www.featuretools.org",
        ],
    )
    correct_urls = [
        "play.google.com",
        "mplay.google.co.in",
        "lplay.google.co.in",
        "play.google.co.in",
        "tplay.google.co.in",
        "google.co.in",
        "google.co.in",
        "google.com",
        "compzets.com",
        "compzets.com",
        "facebook.com",
        "compzets.net",
        "featuretools.org",
    ]
    np.testing.assert_array_equal(url_to_domain(urls), correct_urls)


def test_url_to_domain_long_url():
    url_to_domain = URLToDomain()
    urls = pd.Series(
        [
            "http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart",
        ],
    )
    correct_urls = ["chart.apis.google.com"]
    results = url_to_domain(urls)
    np.testing.assert_array_equal(results, correct_urls)


def test_url_to_domain_nan():
    url_to_domain = URLToDomain()
    urls = pd.Series(["www.featuretools.com", np.nan], dtype="object")
    correct_urls = pd.Series(["featuretools.com", np.nan], dtype="object")
    results = url_to_domain(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_protocol_urls():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(
        [
            "https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22",
            "http://mplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://lplay.google.co.in/sadfask/asdkfals?dk=10",
            "www.google.co.in/sadfask/asdkfals?dk=10",
            "http://user:pass@google.com/?a=b#asdd",
            "https://www.compzets.com?asd=10",
            "www.compzets.com?asd=10",
            "facebook.com",
            "https://www.compzets.net?asd=10",
            "http://www.featuretools.org",
            "https://featuretools.com",
        ],
    )
    correct_urls = pd.Series(
        [
            "https",
            "http",
            "http",
            np.nan,
            "http",
            "https",
            np.nan,
            np.nan,
            "https",
            "http",
            "https",
        ],
    )
    results = url_to_protocol(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_protocol_long_url():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(
        [
            "http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart",
        ],
    )
    correct_urls = ["http"]
    results = url_to_protocol(urls)
    np.testing.assert_array_equal(results, correct_urls)


def test_url_to_protocol_nan():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(["www.featuretools.com", np.nan, ""], dtype="object")
    correct_urls = pd.Series([np.nan, np.nan, np.nan], dtype="object")
    results = url_to_protocol(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_tld_urls():
    url_to_tld = URLToTLD()
    urls = pd.Series(
        [
            "https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22",
            "http://mplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://lplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://play.google.co.in/sadfask/asdkfals?dk=10",
            "http://tplay.google.co.in/sadfask/asdkfals?dk=10",
            "http://www.google.co.in/sadfask/asdkfals?dk=10",
            "www.google.co.in/sadfask/asdkfals?dk=10",
            "http://user:pass@google.com/?a=b#asdd",
            "https://www.compzets.dev?asd=10",
            "www.compzets.com?asd=10",
            "https://www.compzets.net?asd=10",
            "http://www.featuretools.org",
            "featuretools.org",
        ],
    )
    correct_urls = [
        "com",
        "in",
        "in",
        "in",
        "in",
        "in",
        "in",
        "com",
        "dev",
        "com",
        "net",
        "org",
        "org",
    ]
    np.testing.assert_array_equal(url_to_tld(urls), correct_urls)


def test_url_to_tld_long_url():
    url_to_tld = URLToTLD()
    urls = pd.Series(
        [
            "http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart",
        ],
    )
    correct_urls = ["com"]
    np.testing.assert_array_equal(url_to_tld(urls), correct_urls)


def test_url_to_tld_nan():
    url_to_tld = URLToTLD()
    urls = pd.Series(
        ["www.featuretools.com", np.nan, "featuretools", ""],
        dtype="object",
    )
    correct_urls = pd.Series(["com", np.nan, np.nan, np.nan], dtype="object")
    results = url_to_tld(urls)
    pd.testing.assert_series_equal(results, correct_urls, check_names=False)


def test_is_free_email_domain_valid_addresses():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(
        [
            "test@hotmail.com",
            "name@featuretools.com",
            "nobody@yahoo.com",
            "free@gmail.com",
        ],
    )
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([True, False, True, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_valid_addresses_whitespace():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(
        [
            " test@hotmail.com",
            " name@featuretools.com",
            "nobody@yahoo.com ",
            " free@gmail.com ",
        ],
    )
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([True, False, True, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_nan():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([np.nan, "name@featuretools.com", "nobody@yahoo.com"])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, False, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_empty_string():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(["", "name@featuretools.com", "nobody@yahoo.com"])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, False, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_empty_series():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([], dtype="category")
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([], dtype="category")
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_invalid_email():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(
        [
            np.nan,
            "this is not an email address",
            "name@featuretools.com",
            "nobody@yahoo.com",
            1234,
            1.23,
            True,
        ],
    )
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, np.nan, False, True, np.nan, np.nan, np.nan])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_all_nan():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([np.nan, np.nan])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, np.nan], dtype=object)
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_valid_addresses():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series(
        [
            "test@hotmail.com",
            "name@featuretools.com",
            "nobody@yahoo.com",
            "free@gmail.com",
        ],
    )
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series(
        ["hotmail.com", "featuretools.com", "yahoo.com", "gmail.com"],
    )
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_valid_addresses_whitespace():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series(
        [
            " test@hotmail.com",
            " name@featuretools.com",
            "nobody@yahoo.com ",
            " free@gmail.com ",
        ],
    )
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series(
        ["hotmail.com", "featuretools.com", "yahoo.com", "gmail.com"],
    )
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_nan():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([np.nan, "name@featuretools.com", "nobody@yahoo.com"])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, "featuretools.com", "yahoo.com"])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_empty_string():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series(["", "name@featuretools.com", "nobody@yahoo.com"])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, "featuretools.com", "yahoo.com"])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_empty_series():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([], dtype="category")
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([], dtype="category")
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_invalid_email():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series(
        [
            np.nan,
            "this is not an email address",
            "name@featuretools.com",
            "nobody@yahoo.com",
            1234,
            1.23,
            True,
        ],
    )
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series(
        [np.nan, np.nan, "featuretools.com", "yahoo.com", np.nan, np.nan, np.nan],
    )
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_all_nan():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([np.nan, np.nan])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, np.nan], dtype=object)
    pd.testing.assert_series_equal(answers, correct_answers)


def test_trans_primitives_can_init_without_params():
    trans_primitives = get_transform_primitives().values()
    for trans_primitive in trans_primitives:
        trans_primitive()


def test_numeric_lag_future_warning():
    warning_text = "NumericLag is deprecated and will be removed in a future version. Please use the 'Lag' primitive instead."
    with pytest.warns(FutureWarning, match=warning_text):
        NumericLag()


def test_lag_regular():
    primitive_instance = Lag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))

    answer = pd.Series(primitive_func(array, time_array))

    correct_answer = pd.Series([np.nan, 1, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_period():
    primitive_instance = Lag(periods=3)
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))

    answer = pd.Series(primitive_func(array, time_array))

    correct_answer = pd.Series([np.nan, np.nan, np.nan, 1])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_negative_period():
    primitive_instance = Lag(periods=-2)
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))

    answer = pd.Series(primitive_func(array, time_array))

    correct_answer = pd.Series([3, 4, np.nan, np.nan])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_starts_with_nan():
    primitive_instance = Lag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([np.nan, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))

    answer = pd.Series(primitive_func(array, time_array))

    correct_answer = pd.Series([np.nan, np.nan, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_ends_with_nan():
    primitive_instance = Lag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, np.nan])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))

    answer = pd.Series(primitive_func(array, time_array))

    correct_answer = pd.Series([np.nan, 1, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


@pytest.mark.parametrize(
    "input_array,expected_output",
    [
        (
            pd.Series(["hello", "world", "foo", "bar"], dtype="string"),
            pd.Series([np.nan, "hello", "world", "foo"], dtype="string"),
        ),
        (
            pd.Series(["cow", "cow", "pig", "pig"], dtype="category"),
            pd.Series([np.nan, "cow", "cow", "pig"], dtype="category"),
        ),
        (
            pd.Series([True, False, True, False], dtype="bool"),
            pd.Series([np.nan, True, False, True], dtype="object"),
        ),
        (
            pd.Series([True, False, True, False], dtype="boolean"),
            pd.Series([np.nan, True, False, True], dtype="boolean"),
        ),
        (
            pd.Series([1.23, 2.45, 3.56, 4.98], dtype="float"),
            pd.Series([np.nan, 1.23, 2.45, 3.56], dtype="float"),
        ),
        (
            pd.Series([1, 2, 3, 4], dtype="Int64"),
            pd.Series([np.nan, 1, 2, 3], dtype="Int64"),
        ),
        (
            pd.Series([1, 2, 3, 4], dtype="int64"),
            pd.Series([np.nan, 1, 2, 3], dtype="float64"),
        ),
    ],
)
def test_lag_with_different_dtypes(input_array, expected_output):
    primitive_instance = Lag()
    primitive_func = primitive_instance.get_function()
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq="D"))
    answer = pd.Series(primitive_func(input_array, time_array))
    pd.testing.assert_series_equal(answer, expected_output)


def test_date_to_time_zone_primitive():
    primitive_func = DateToTimeZone().get_function()
    x = pd.Series(
        [
            datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
            datetime(2010, 1, 10, tzinfo=timezone("Singapore")),
            datetime(2020, 1, 1, tzinfo=timezone("UTC")),
            datetime(2010, 1, 1, tzinfo=timezone("Europe/London")),
        ],
    )
    answer = pd.Series(["America/Los_Angeles", "Singapore", "UTC", "Europe/London"])
    pd.testing.assert_series_equal(primitive_func(x), answer)


def test_date_to_time_zone_datetime64():
    primitive_func = DateToTimeZone().get_function()
    x = pd.Series(
        [
            datetime(2010, 1, 1),
            datetime(2010, 1, 10),
            datetime(2020, 1, 1),
        ],
    ).astype("datetime64[ns]")
    x = x.dt.tz_localize("America/Los_Angeles")
    answer = pd.Series(["America/Los_Angeles"] * 3)
    pd.testing.assert_series_equal(primitive_func(x), answer)


def test_date_to_time_zone_naive_dates():
    primitive_func = DateToTimeZone().get_function()
    x = pd.Series(
        [
            datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
            datetime(2010, 1, 1),
            datetime(2010, 1, 2),
        ],
    )
    answer = pd.Series(["America/Los_Angeles", np.nan, np.nan])
    pd.testing.assert_series_equal(primitive_func(x), answer)


def test_date_to_time_zone_nan():
    primitive_func = DateToTimeZone().get_function()
    x = pd.Series(
        [
            datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
            pd.NaT,
            np.nan,
        ],
    )
    answer = pd.Series(["America/Los_Angeles", np.nan, np.nan])
    pd.testing.assert_series_equal(primitive_func(x), answer)


def test_rate_of_change_primitive_regular_interval():
    rate_of_change = RateOfChange()
    times = pd.date_range(start="2019-01-01", freq="2s", periods=5)
    values = [0, 30, 180, -90, 0]
    expected = pd.Series([np.nan, 15, 75, -135, 45])
    actual = rate_of_change(values, times)
    pd.testing.assert_series_equal(actual, expected)


def test_rate_of_change_primitive_uneven_interval():
    rate_of_change = RateOfChange()
    times = pd.to_datetime(
        [
            "2019-01-01 00:00:00",
            "2019-01-01 00:00:01",
            "2019-01-01 00:00:03",
            "2019-01-01 00:00:07",
            "2019-01-01 00:00:08",
        ],
    )
    values = [0, 30, 180, -90, 0]
    expected = pd.Series([np.nan, 30, 75, -67.5, 90])
    actual = rate_of_change(values, times)
    pd.testing.assert_series_equal(actual, expected)


def test_rate_of_change_primitive_with_nan():
    rate_of_change = RateOfChange()
    times = pd.date_range(start="2019-01-01", freq="2s", periods=5)
    values = [0, 30, np.nan, -90, 0]
    expected = pd.Series([np.nan, 15, np.nan, np.nan, 45])
    actual = rate_of_change(values, times)
    pd.testing.assert_series_equal(actual, expected)


class TestFileExtension(PrimitiveTestBase):
    primitive = FileExtension

    def test_filepaths(self):
        primitive_func = FileExtension().get_function()
        array = pd.Series(
            [
                "doc.txt",
                "~/documents/data.json",
                "data.JSON",
                "C:\\Projects\\apilibrary\\apilibrary.sln",
            ],
            dtype="string",
        )
        answer = pd.Series([".txt", ".json", ".json", ".sln"], dtype="string")
        pd.testing.assert_series_equal(primitive_func(array), answer)

    def test_invalid(self):
        primitive_func = FileExtension().get_function()
        array = pd.Series(["doc.txt", "~/documents/data", np.nan], dtype="string")
        answer = pd.Series([".txt", np.nan, np.nan], dtype="string")
        pd.testing.assert_series_equal(primitive_func(array), answer)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(
            es,
            aggregation,
            transform,
            self.primitive,
            target_dataframe_name="sessions",
        )


class TestIsFirstWeekOfMonth(PrimitiveTestBase):
    primitive = IsFirstWeekOfMonth

    def test_valid_dates(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                pd.to_datetime("03/03/2019"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array).tolist()
        correct_answers = [True, False, False, False]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_leap_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                pd.to_datetime("02/29/2016"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array).tolist()
        correct_answers = [True, False, False, False]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_year_before_1970(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("06/01/1965"),
                pd.to_datetime("03/02/2019"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array).tolist()
        correct_answers = [True, True, False, False]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_year_after_2038(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("12/31/2040"),
                pd.to_datetime("01/01/2040"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array).tolist()
        correct_answers = [False, True, False, False]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_nan_input(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                np.nan,
                np.datetime64("NaT"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array).tolist()
        correct_answers = [True, np.nan, np.nan, False]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)


class TestNthWeekOfMonth(PrimitiveTestBase):
    primitive = NthWeekOfMonth

    def test_valid_dates(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                pd.to_datetime("03/03/2019"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
                pd.to_datetime("09/01/2019"),
            ],
        )
        answers = primitive_func(array)
        correct_answers = [1, 2, 6, 5, 1]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_leap_year(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                pd.to_datetime("02/29/2016"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array)
        correct_answers = [1, 5, 6, 5]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_year_before_1970(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("06/06/1965"),
                pd.to_datetime("03/02/2019"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array)
        correct_answers = [2, 1, 6, 5]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_year_after_2038(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("12/31/2040"),
                pd.to_datetime("01/01/2001"),
                pd.to_datetime("03/31/2019"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array)
        correct_answers = [6, 1, 6, 5]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_nan_input(self):
        primitive_func = self.primitive().get_function()
        array = pd.Series(
            [
                pd.to_datetime("03/01/2019"),
                np.nan,
                np.datetime64("NaT"),
                pd.to_datetime("03/30/2019"),
            ],
        )
        answers = primitive_func(array)
        correct_answers = [1, np.nan, np.nan, 5]
        np.testing.assert_array_equal(answers, correct_answers)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
