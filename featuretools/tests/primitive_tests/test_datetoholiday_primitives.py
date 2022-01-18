from datetime import datetime
import pandas as pd
import numpy as np

from featuretools.primitives.standard.transform_primitive import DateToHoliday


def test_datetoholiday():
    date_to_holiday = DateToHoliday()

    dates = pd.Series([
        datetime(2016, 1, 1),
        datetime(2016, 2, 27),
        datetime(2017, 5, 29, 10, 30, 5),
        datetime(2018, 7, 4)])

    holiday_series = date_to_holiday(dates).tolist()

    assert holiday_series[0] == "New Year's Day"
    assert np.isnan(holiday_series[1])
    assert holiday_series[2] == "Memorial Day"
    assert holiday_series[3] == "Independence Day"
