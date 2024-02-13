from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import DateToHoliday


def test_datetoholiday():
    date_to_holiday = DateToHoliday()

    dates = pd.Series(
        [
            datetime(2016, 1, 1),
            datetime(2016, 2, 27),
            datetime(2017, 5, 29, 10, 30, 5),
            datetime(2018, 7, 4),
        ],
    )

    holiday_series = date_to_holiday(dates).tolist()

    assert holiday_series[0] == "New Year's Day"
    assert np.isnan(holiday_series[1])
    assert holiday_series[2] == "Memorial Day"
    assert holiday_series[3] == "Independence Day"


def test_datetoholiday_error():
    error_text = r"must be one of the available countries.*"
    with pytest.raises(ValueError, match=error_text):
        DateToHoliday(country="UNK")


def test_nat():
    date_to_holiday = DateToHoliday()
    case = pd.Series(
        [
            "2019-10-14",
            "NaT",
            "2016-02-15",
            "NaT",
        ],
    ).astype("datetime64[ns]")
    answer = ["Columbus Day", np.nan, "Washington's Birthday", np.nan]
    given_answer = date_to_holiday(case).astype("str")
    np.testing.assert_array_equal(given_answer, answer)


def test_valid_country():
    date_to_holiday = DateToHoliday(country="Canada")
    case = pd.Series(
        [
            "2016-07-01",
            "2016-11-11",
            "2018-12-25",
        ],
    ).astype("datetime64[ns]")
    answer = ["Canada Day", np.nan, "Christmas Day"]
    given_answer = date_to_holiday(case).astype("str")
    np.testing.assert_array_equal(given_answer, answer)


def test_multiple_countries():
    dth_mexico = DateToHoliday(country="Mexico")

    case = pd.Series([datetime(2000, 9, 16), datetime(2005, 1, 1)])
    assert len(dth_mexico(case)) > 1

    dth_india = DateToHoliday(country="IND")
    case = pd.Series([datetime(2048, 1, 1), datetime(2048, 10, 2)])
    assert len(dth_india(case)) > 1

    dth_uk = DateToHoliday(country="UK")
    case = pd.Series([datetime(2048, 3, 17), datetime(2048, 4, 6)])
    assert len(dth_uk(case)) > 1

    countries = [
        "Argentina",
        "AU",
        "Austria",
        "BY",
        "Belgium",
        "Brazil",
        "Canada",
        "Colombia",
        "Croatia",
        "England",
        "Finland",
        "FRA",
        "Germany",
        "Germany",
        "Italy",
        "NewZealand",
        "PortugalExt",
        "PTE",
        "Spain",
        "ES",
        "Switzerland",
        "UnitedStates",
        "US",
        "UK",
        "UA",
        "CH",
        "SE",
        "ZA",
    ]
    for x in countries:
        DateToHoliday(country=x)


def test_with_timezone_aware_datetimes():
    df = pd.DataFrame(
        {
            "non_timezone_aware_with_time": pd.date_range(
                "2018-07-03 09:00",
                periods=3,
            ),
            "non_timezone_aware_no_time": pd.date_range("2018-07-03", periods=3),
            "timezone_aware_with_time": pd.date_range(
                "2018-07-03 09:00",
                periods=3,
            ).tz_localize(tz="US/Eastern"),
            "timezone_aware_no_time": pd.date_range(
                "2018-07-03",
                periods=3,
            ).tz_localize(tz="US/Eastern"),
        },
    )

    date_to_holiday = DateToHoliday(country="US")
    expected = [np.nan, "Independence Day", np.nan]
    for col in df.columns:
        actual = date_to_holiday(df[col]).astype("str")
        np.testing.assert_array_equal(actual, expected)
