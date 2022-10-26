from datetime import datetime

import holidays
import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import DistanceToHoliday


def test_distanceholiday():
    distance_to_holiday = DistanceToHoliday("New Year's Day")
    dates = pd.Series(
        [
            datetime(2010, 1, 1),
            datetime(2012, 5, 31),
            datetime(2017, 7, 31),
            datetime(2020, 12, 31),
        ],
    )

    expected = [0, -151, 154, 1]
    output = distance_to_holiday(dates).tolist()
    np.testing.assert_array_equal(output, expected)


def test_holiday_out_of_range():
    date_to_holiday = DistanceToHoliday("Boxing Day", country="Canada")

    array = pd.Series(
        [
            datetime(2010, 1, 1),
            datetime(2012, 5, 31),
            datetime(2017, 7, 31),
            datetime(2020, 12, 31),
        ],
    )
    days_to_boxing_day = -157 if float(holidays.__version__) >= 0.15 else 209
    answer = pd.Series([np.nan, days_to_boxing_day, 148, np.nan])
    pd.testing.assert_series_equal(date_to_holiday(array), answer, check_names=False)


def test_unknown_country_error():
    error_text = r"must be one of the available countries.*"
    with pytest.raises(ValueError, match=error_text):
        DistanceToHoliday("Victoria Day", country="UNK")


def test_unknown_holiday_error():
    error_text = r"must be one of the available holidays.*"
    with pytest.raises(ValueError, match=error_text):
        DistanceToHoliday("Alteryx Day")


def test_nat():
    date_to_holiday = DistanceToHoliday("New Year's Day")
    case = pd.Series(
        [
            "2010-01-01",
            "NaT",
            "2012-05-31",
            "NaT",
        ],
    ).astype("datetime64")
    answer = [0, np.nan, -151, np.nan]
    given_answer = date_to_holiday(case).astype("float")
    np.testing.assert_array_equal(given_answer, answer)


def test_valid_country():
    distance_to_holiday = DistanceToHoliday("Victoria Day", country="Canada")
    case = pd.Series(
        [
            "2010-01-01",
            "2012-05-31",
            "2017-07-31",
            "2020-12-31",
        ],
    ).astype("datetime64")
    answer = [143, -10, -70, 144]
    given_answer = distance_to_holiday(case).astype("float")
    np.testing.assert_array_equal(given_answer, answer)


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

    distance_to_holiday = DistanceToHoliday("Independence Day", country="US")
    expected = [1, 0, -1]
    for col in df.columns:
        actual = distance_to_holiday(df[col])
        np.testing.assert_array_equal(actual, expected)
