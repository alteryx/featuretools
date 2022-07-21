import holidays
import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    AgeFractional,
    BooleanNullable,
    Categorical,
    Datetime,
    Ordinal,
)

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.primitives.utils import HolidayUtil
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library


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
        try:
            self.holidays = holidays.country_holidays(country=self.country)
        except NotImplementedError:
            available_countries = (
                "https://github.com/dr-prodigy/python-holidays#available-countries"
            )
            error = "must be one of the available countries:\n%s" % available_countries
            raise ValueError(error)
        years_list = [1950 + x for x in range(150)]
        self.federal_holidays = getattr(holidays, country)(years=years_list)

    def get_function(self):
        def is_federal_holiday(x):
            holidays_df = pd.DataFrame(
                sorted(self.federal_holidays.items()),
                columns=["dates", "names"],
            )
            is_holiday = x.dt.normalize().isin(holidays_df.dates)
            if x.isnull().values.any():
                is_holiday = is_holiday.astype("object")
                is_holiday[x.isnull()] = np.nan
            return is_holiday.values

        return is_federal_holiday
