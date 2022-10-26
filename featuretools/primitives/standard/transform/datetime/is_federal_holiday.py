import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.datetime.utils import HolidayUtil


class IsFederalHoliday(TransformPrimitive):
    """Determines if a given datetime is a federal holiday.

    Description:
        This primtive currently only works for the United States
        and Canada with dates between 1950 and 2100.

    Args:
        country (str): Country to use for determining Holidays.
            Default is 'US'. Should be one of the available countries here:
            https://github.com/dr-prodigy/python-holidays#available-countries

    Examples:
        >>> from datetime import datetime
        >>> is_federal_holiday = IsFederalHoliday(country="US")
        >>> is_federal_holiday([
        ...     datetime(2019, 7, 4, 10, 0, 30),
        ...     datetime(2019, 2, 26)]).tolist()
        [True, False]
    """

    name = "is_federal_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)

    def __init__(self, country="US"):
        self.country = country
        self.holidayUtil = HolidayUtil(country)

    def get_function(self):
        def is_federal_holiday(x):
            holidays_df = self.holidayUtil.to_df()
            is_holiday = x.dt.normalize().isin(holidays_df.holiday_date)
            if x.isnull().values.any():
                is_holiday = is_holiday.astype("object")
                is_holiday[x.isnull()] = np.nan
            return is_holiday.values

        return is_federal_holiday
