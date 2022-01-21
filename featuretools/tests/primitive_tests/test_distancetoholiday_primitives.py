from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives.standard.transform_primitive import (
    DistanceToHoliday
)


def test_distanceholiday():
    distance_to_holiday = DistanceToHoliday("New Year's Day")
    dates = [datetime(2010, 1, 1),
             datetime(2012, 5, 31),
             datetime(2017, 7, 31),
             datetime(2020, 12, 31)]

    distance_list = distance_to_holiday(dates).tolist()

    assert distance_list[0] == 0
    assert distance_list[1] == -151
    assert distance_list[2] == 154
    assert distance_list[3] == 1


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
