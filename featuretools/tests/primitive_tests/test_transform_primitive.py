from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import (
    Age,
    EmailAddressToDomain,
    IsFreeEmailDomain,
    NumericLag,
    TimeSince,
    URLToDomain,
    URLToProtocol,
    URLToTLD,
    Week,
    get_transform_primitives
)


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


def test_url_to_domain_urls():
    url_to_domain = URLToDomain()
    urls = pd.Series(['https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22',
                      'http://mplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://lplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://play.google.co.in/sadfask/asdkfals?dk=10',
                      'http://tplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://www.google.co.in/sadfask/asdkfals?dk=10',
                      'www.google.co.in/sadfask/asdkfals?dk=10',
                      'http://user:pass@google.com/?a=b#asdd',
                      'https://www.compzets.com?asd=10',
                      'www.compzets.com?asd=10',
                      'facebook.com',
                      'https://www.compzets.net?asd=10',
                      'http://www.featuretools.org'])
    correct_urls = ['play.google.com',
                    'mplay.google.co.in',
                    'lplay.google.co.in',
                    'play.google.co.in',
                    'tplay.google.co.in',
                    'google.co.in',
                    'google.co.in',
                    'google.com',
                    'compzets.com',
                    'compzets.com',
                    'facebook.com',
                    'compzets.net',
                    'featuretools.org']
    np.testing.assert_array_equal(url_to_domain(urls), correct_urls)


def test_url_to_domain_long_url():
    url_to_domain = URLToDomain()
    urls = pd.Series(["http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart"])
    correct_urls = ['chart.apis.google.com']
    results = url_to_domain(urls)
    np.testing.assert_array_equal(results, correct_urls)


def test_url_to_domain_nan():
    url_to_domain = URLToDomain()
    urls = pd.Series(['www.featuretools.com', np.nan], dtype='object')
    correct_urls = pd.Series(['featuretools.com', np.nan], dtype='object')
    results = url_to_domain(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_protocol_urls():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(['https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22',
                      'http://mplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://lplay.google.co.in/sadfask/asdkfals?dk=10',
                      'www.google.co.in/sadfask/asdkfals?dk=10',
                      'http://user:pass@google.com/?a=b#asdd',
                      'https://www.compzets.com?asd=10',
                      'www.compzets.com?asd=10',
                      'facebook.com',
                      'https://www.compzets.net?asd=10',
                      'http://www.featuretools.org',
                      'https://featuretools.com'])
    correct_urls = pd.Series(['https',
                              'http',
                              'http',
                              np.nan,
                              'http',
                              'https',
                              np.nan,
                              np.nan,
                              'https',
                              'http',
                              'https'])
    results = url_to_protocol(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_protocol_long_url():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(["http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart"])
    correct_urls = ['http']
    results = url_to_protocol(urls)
    np.testing.assert_array_equal(results, correct_urls)


def test_url_to_protocol_nan():
    url_to_protocol = URLToProtocol()
    urls = pd.Series(['www.featuretools.com', np.nan, ''], dtype='object')
    correct_urls = pd.Series([np.nan, np.nan, np.nan], dtype='object')
    results = url_to_protocol(urls)
    pd.testing.assert_series_equal(results, correct_urls)


def test_url_to_tld_urls():
    url_to_tld = URLToTLD()
    urls = pd.Series(['https://play.google.com/store/apps/details?id=com.skgames.trafficracer%22',
                      'http://mplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://lplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://play.google.co.in/sadfask/asdkfals?dk=10',
                      'http://tplay.google.co.in/sadfask/asdkfals?dk=10',
                      'http://www.google.co.in/sadfask/asdkfals?dk=10',
                      'www.google.co.in/sadfask/asdkfals?dk=10',
                      'http://user:pass@google.com/?a=b#asdd',
                      'https://www.compzets.dev?asd=10',
                      'www.compzets.com?asd=10',
                      'https://www.compzets.net?asd=10',
                      'http://www.featuretools.org',
                      'featuretools.org'])
    correct_urls = ['com',
                    'in',
                    'in',
                    'in',
                    'in',
                    'in',
                    'in',
                    'com',
                    'dev',
                    'com',
                    'net',
                    'org',
                    'org']
    np.testing.assert_array_equal(url_to_tld(urls), correct_urls)


def test_url_to_tld_long_url():
    url_to_tld = URLToTLD()
    urls = pd.Series(["http://chart.apis.google.com/chart?chs=500x500&chma=0,0,100, \
                        100&cht=p&chco=FF0000%2CFFFF00%7CFF8000%2C00FF00%7C00FF00%2C0 \
                        000FF&chd=t%3A122%2C42%2C17%2C10%2C8%2C7%2C7%2C7%2C7%2C6%2C6% \
                        2C6%2C6%2C5%2C5&chl=122%7C42%7C17%7C10%7C8%7C7%7C7%7C7%7C7%7C \
                        6%7C6%7C6%7C6%7C5%7C5&chdl=android%7Cjava%7Cstack-trace%7Cbro \
                        adcastreceiver%7Candroid-ndk%7Cuser-agent%7Candroid-webview%7 \
                        Cwebview%7Cbackground%7Cmultithreading%7Candroid-source%7Csms \
                        %7Cadb%7Csollections%7Cactivity|Chart"])
    correct_urls = ['com']
    np.testing.assert_array_equal(url_to_tld(urls), correct_urls)


def test_url_to_tld_nan():
    url_to_tld = URLToTLD()
    urls = pd.Series(['www.featuretools.com', np.nan, 'featuretools', ''], dtype='object')
    correct_urls = pd.Series(['com', np.nan, np.nan, np.nan], dtype='object')
    results = url_to_tld(urls)
    pd.testing.assert_series_equal(results, correct_urls, check_names=False)


def test_is_free_email_domain_valid_addresses():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(['test@hotmail.com', 'name@featuretools.com', 'nobody@yahoo.com', 'free@gmail.com'])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([True, False, True, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_valid_addresses_whitespace():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([' test@hotmail.com', ' name@featuretools.com', 'nobody@yahoo.com ', ' free@gmail.com '])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([True, False, True, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_nan():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([np.nan, 'name@featuretools.com', 'nobody@yahoo.com'])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, False, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_empty_string():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series(['', 'name@featuretools.com', 'nobody@yahoo.com'])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([np.nan, False, True])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_empty_series():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([])
    answers = pd.Series(is_free_email_domain(array))
    correct_answers = pd.Series([])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_is_free_email_domain_invalid_email():
    is_free_email_domain = IsFreeEmailDomain()
    array = pd.Series([np.nan, 'this is not an email address', 'name@featuretools.com', 'nobody@yahoo.com', 1234, 1.23, True])
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
    array = pd.Series(['test@hotmail.com', 'name@featuretools.com', 'nobody@yahoo.com', 'free@gmail.com'])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series(['hotmail.com', 'featuretools.com', 'yahoo.com', 'gmail.com'])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_valid_addresses_whitespace():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([' test@hotmail.com', ' name@featuretools.com', 'nobody@yahoo.com ', ' free@gmail.com '])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series(['hotmail.com', 'featuretools.com', 'yahoo.com', 'gmail.com'])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_nan():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([np.nan, 'name@featuretools.com', 'nobody@yahoo.com'])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, 'featuretools.com', 'yahoo.com'])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_empty_string():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series(['', 'name@featuretools.com', 'nobody@yahoo.com'])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, 'featuretools.com', 'yahoo.com'])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_empty_series():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([])
    pd.testing.assert_series_equal(answers, correct_answers)


def test_email_address_to_domain_invalid_email():
    email_address_to_domain = EmailAddressToDomain()
    array = pd.Series([np.nan, 'this is not an email address', 'name@featuretools.com', 'nobody@yahoo.com', 1234, 1.23, True])
    answers = pd.Series(email_address_to_domain(array))
    correct_answers = pd.Series([np.nan, np.nan, 'featuretools.com', 'yahoo.com', np.nan, np.nan, np.nan])
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


def test_lag_regular():
    primitive_instance = NumericLag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([np.nan, 1, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_period():
    primitive_instance = NumericLag(periods=3)
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([np.nan, np.nan, np.nan, 1])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_negative_period():
    primitive_instance = NumericLag(periods=-2)
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([3, 4, np.nan, np.nan])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_fill_value():
    primitive_instance = NumericLag(fill_value=10)
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([10, 1, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_starts_with_nan():
    primitive_instance = NumericLag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([np.nan, 2, 3, 4])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([np.nan, np.nan, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_lag_ends_with_nan():
    primitive_instance = NumericLag()
    primitive_func = primitive_instance.get_function()

    array = pd.Series([1, 2, 3, np.nan])
    time_array = pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D'))

    answer = pd.Series(primitive_func(time_array, array))

    correct_answer = pd.Series([np.nan, 1, 2, 3])
    pd.testing.assert_series_equal(answer, correct_answer)
