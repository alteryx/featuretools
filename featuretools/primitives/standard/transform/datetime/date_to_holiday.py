import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.datetime.utils import HolidayUtil


class DateToHoliday(TransformPrimitive):
    """Transforms time of an instance into the holiday name, if there is one.

    Description:
        If there is no holiday, it returns `NaN`. Currently only works for the
        United States and Canada with dates between 1950 and 2100.

    Args:
        country (str): Country to use for determining Holidays.
            Default is 'US'. Should be one of the available countries here:
            https://github.com/dr-prodigy/python-holidays#available-countries

    Examples:
        >>> from datetime import datetime
        >>> date_to_holiday = DateToHoliday()
        >>> dates = pd.Series([datetime(2016, 1, 1),
        ...          datetime(2016, 2, 27),
        ...          datetime(2017, 5, 29, 10, 30, 5),
        ...          datetime(2018, 7, 4)])
        >>> date_to_holiday(dates).tolist()
        ["New Year's Day", nan, 'Memorial Day', 'Independence Day']

        We can also change the country.

        >>> date_to_holiday_canada = DateToHoliday(country='Canada')
        >>> dates = pd.Series([datetime(2016, 7, 1),
        ...          datetime(2016, 11, 15),
        ...          datetime(2018, 12, 25)])
        >>> date_to_holiday_canada(dates).tolist()
        ['Canada Day', nan, 'Christmas Day']
    """

    name = "date_to_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def __init__(self, country="US"):
        self.country = country
        self.holidayUtil = HolidayUtil(country)

    def get_function(self):
        def date_to_holiday(x):
            holiday_df = self.holidayUtil.to_df()
            df = pd.DataFrame({"date": x})
            df["date"] = df["date"].dt.date.astype("datetime64[ns]")

            df = df.merge(
                holiday_df,
                how="left",
                left_on="date",
                right_on="holiday_date",
            )
            return df.names.values

        return date_to_holiday
