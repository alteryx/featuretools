from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from featuretools.primitives.standard.transform_primitive import DistanceToHoliday


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


def test_distanceholiday_error():
    error_text = r"must be one of the available countries.*"
    with pytest.raises(ValueError, match=error_text):
        DistanceToHoliday("Victoria Day", country='UNK')
