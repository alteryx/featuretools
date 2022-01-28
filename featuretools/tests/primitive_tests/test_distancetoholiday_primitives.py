from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives.standard.datetime_transform_primitives import (
    DistanceToHoliday
)


def test_distanceholiday():
    distance_to_holiday = DistanceToHoliday("New Year's Day")
    dates = pd.Series([datetime(2010, 1, 1),
                       datetime(2012, 5, 31),
                       datetime(2017, 7, 31),
                       datetime(2020, 12, 31)])

    expected = [0, -151, 154, 1]
    output = distance_to_holiday(dates).tolist()
    np.testing.assert_array_equal(output, expected)


def test_holiday_out_of_range():
    date_to_holiday = DistanceToHoliday("Boxing Day", country='Canada')

    array = pd.Series([datetime(2010, 1, 1),
                       datetime(2012, 5, 31),
                       datetime(2017, 7, 31),
                       datetime(2020, 12, 31)])
    answer = pd.Series([np.nan, 209, 148, np.nan])
    pd.testing.assert_series_equal(
        date_to_holiday(array),
        answer,
        check_names=False
    )


def test_unknown_country_error():
    error_text = r"must be one of the available countries.*"
    with pytest.raises(ValueError, match=error_text):
        DistanceToHoliday("Victoria Day", country='UNK')


def test_unknown_holiday_error():
    error_text = r"must be one of the available holidays.*"
    with pytest.raises(ValueError, match=error_text):
        DistanceToHoliday("Alteryx Day")


def test_nat():
    date_to_holiday = DistanceToHoliday("New Year's Day")
    case = pd.Series([
        '2010-01-01',
        'NaT',
        '2012-05-31',
        'NaT',
    ]).astype('datetime64')
    answer = [0, np.nan, -151, np.nan]
    given_answer = date_to_holiday(case).astype('float')
    np.testing.assert_array_equal(given_answer, answer)


def test_valid_country():
    distance_to_holiday = DistanceToHoliday("Victoria Day", country='Canada')
    case = pd.Series([
        '2010-01-01',
        '2012-05-31',
        '2017-07-31',
        '2020-12-31',
    ]).astype('datetime64')
    answer = [143, -10, -70, 144]
    given_answer = distance_to_holiday(case).astype('float')
    np.testing.assert_array_equal(given_answer, answer)
