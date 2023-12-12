import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.datetime.utils import HolidayUtil


class DistanceToHoliday(TransformPrimitive):
    """Computes the number of days before or after a given holiday.

    Description:
        For a list of dates, return the distance from the nearest
        occurrence of a chosen holiday. The distance is returned in
        days. If the closest occurrence is prior to the date given,
        return a negative number.

        If a date is missing, return `NaN`.

        Currently only works with dates between 1950 and 2100.

    Args:
        holiday (str): Name of the holiday. Defaults to New Year's Day.

        country (str): Specifies which country's calendar to use for the
            given holiday. Default is `US`.

    Examples:
        >>> from datetime import datetime
        >>> distance_to_holiday = DistanceToHoliday("New Year's Day")
        >>> dates = [datetime(2010, 1, 1),
        ...          datetime(2012, 5, 31),
        ...          datetime(2017, 7, 31),
        ...          datetime(2020, 12, 31)]
        >>> distance_to_holiday(dates).tolist()
        [0, -151, 154, 1]

        We can also control the country in which we're searching for
            a holiday.

        >>> distance_to_holiday = DistanceToHoliday("Canada Day", country='Canada')
        >>> dates = [datetime(2010, 1, 1),
        ...          datetime(2012, 5, 31),
        ...          datetime(2017, 7, 31),
        ...          datetime(2020, 12, 31)]
        >>> distance_to_holiday(dates).tolist()
        [181, 31, -30, 182]
    """

    name = "distance_to_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    default_value = 0

    def __init__(self, holiday="New Year's Day", country="US"):
        self.country = country
        self.holiday = holiday
        self.holidayUtil = HolidayUtil(country)

        available_holidays = list(set(self.holidayUtil.federal_holidays.values()))
        if self.holiday not in available_holidays:
            error = "must be one of the available holidays:\n%s" % available_holidays
            raise ValueError(error)

    def get_function(self):
        def distance_to_holiday(x):
            holiday_df = self.holidayUtil.to_df()
            holiday_df = holiday_df[holiday_df.names == self.holiday]

            df = pd.DataFrame({"date": x})
            df["x_index"] = df.index  # store original index as a column
            df = df.dropna()
            df = df.sort_values("date")
            df["date"] = df["date"].dt.date.astype("datetime64[ns]")

            matches = pd.merge_asof(
                df,
                holiday_df,
                left_on="date",
                right_on="holiday_date",
                direction="nearest",
                tolerance=pd.Timedelta("365d"),
            )
            matches = matches.set_index("x_index")
            matches["days_diff"] = (matches.holiday_date - matches.date).dt.days

            return matches.days_diff.reindex_like(x)

        return distance_to_holiday
