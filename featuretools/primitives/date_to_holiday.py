import holidays
import pandas as pd
from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime


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

        >>> date_to_holiday_cananda = DateToHoliday(country='Canada')
        >>> dates = pd.Series([datetime(2016, 7, 1),
        ...          datetime(2016, 11, 15),
        ...          datetime(2017, 12, 26),
        ...          datetime(2018, 9, 3)])
        >>> date_to_holiday_cananda(dates).tolist()
        ['Canada Day', nan, 'Boxing Day', 'Labour Day']
    """
    name = "date_to_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def __init__(self, country='US'):
        self.country = country
        try:
            self.holidays = holidays.CountryHoliday(self.country)
        except KeyError:
            available_countries = 'https://github.com/dr-prodigy/python-holidays#available-countries'
            error = 'must be one of the available countries:\n%s' % available_countries
            raise ValueError(error)
        years_list = [1950 + x for x in range(150)]
        self.federal_holidays = getattr(holidays, country)(years=years_list)

    def get_function(self):
        def date_to_holiday(x):
            holidays_df = pd.DataFrame(sorted(self.federal_holidays.items()),
                                       columns=['dates', 'names'])
            holidays_df.dates = holidays_df.dates.astype('datetime64')
            df = pd.DataFrame({'dates': x})
            df.dates = df.dates.dt.normalize().astype('datetime64')
            df = df.merge(holidays_df, how='left')
            return df.names.values
        return date_to_holiday