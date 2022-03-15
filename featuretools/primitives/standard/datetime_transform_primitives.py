import warnings

import holidays
import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    AgeFractional,
    BooleanNullable,
    Categorical,
    Datetime,
    Ordinal
)

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.utils import HolidayUtil
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library


class Age(TransformPrimitive):
    """Calculates the age in years as a floating point number given a
       date of birth.

    Description:
        Age in years is computed by calculating the number of days between
        the date of birth and the reference time and dividing the result
        by 365.

    Examples:
        Determine the age of three people as of Jan 1, 2019
        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age = Age()
        >>> input_ages = [pd.to_datetime("01-01-2000"),
        ...               pd.to_datetime("05-30-1983"),
        ...               pd.to_datetime("10-17-1997")]
        >>> age(input_ages, time=reference_date).tolist()
        [19.013698630136986, 35.61643835616438, 21.221917808219178]
    """
    name = "age"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'date_of_birth'})]
    return_type = ColumnSchema(logical_type=AgeFractional, semantic_tags={'numeric'})
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the age from {}"

    def get_function(self):
        def age(x, time=None):
            return (time - x).dt.days / 365
        return age


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
        ...          datetime(2017, 12, 26),
        ...          datetime(2018, 9, 3)])
        >>> date_to_holiday_canada(dates).tolist()
        ['Canada Day', nan, 'Boxing Day', 'Labour Day']
    """
    name = "date_to_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def __init__(self, country='US'):
        self.country = country
        self.holidayUtil = HolidayUtil(country)

    def get_function(self):
        def date_to_holiday(x):
            holiday_df = self.holidayUtil.to_df()
            df = pd.DataFrame({'date': x})
            df.date = df.date.dt.normalize().astype('datetime64')

            df = df.merge(holiday_df, how='left', left_on='date', right_on='holiday_date')
            return df.names.values
        return date_to_holiday


class Day(TransformPrimitive):
    """Determines the day of the month from a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3),
        ...          datetime(2019, 3, 31)]
        >>> day = Day()
        >>> day(dates).tolist()
        [1, 3, 31]
    """
    name = "day"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 32))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the day of the month of {}"

    def get_function(self):
        def day(vals):
            return vals.dt.day
        return day


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

        >>> distance_to_holiday = DistanceToHoliday("Victoria Day", country='Canada')
        >>> dates = [datetime(2010, 1, 1),
        ...          datetime(2012, 5, 31),
        ...          datetime(2017, 7, 31),
        ...          datetime(2020, 12, 31)]
        >>> distance_to_holiday(dates).tolist()
        [143, -10, -70, 144]
    """
    name = "distance_to_holiday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    default_value = 0

    def __init__(self, holiday="New Year's Day", country="US"):
        self.country = country
        self.holiday = holiday
        self.holidayUtil = HolidayUtil(country)

        available_holidays = list(set(self.holidayUtil.federal_holidays.values()))
        if self.holiday not in available_holidays:
            error = 'must be one of the available holidays:\n%s' % available_holidays
            raise ValueError(error)

    def get_function(self):
        def distance_to_holiday(x):
            holiday_df = self.holidayUtil.to_df()
            holiday_df = holiday_df[holiday_df.names == self.holiday]

            df = pd.DataFrame({'date': x})
            df['x_index'] = df.index  # store original index as a column
            df = df.dropna()
            df = df.sort_values('date')
            df.date = df.date.dt.normalize()

            matches = pd.merge_asof(df, holiday_df, left_on='date', right_on='holiday_date',
                                    direction='nearest', tolerance=pd.Timedelta('365d'))
            matches = matches.set_index('x_index')
            matches['days_diff'] = (matches.holiday_date - matches.date).dt.days

            return matches.days_diff.reindex_like(x)
        return distance_to_holiday


class Hour(TransformPrimitive):
    """Determines the hour value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> hour = Hour()
        >>> hour(dates).tolist()
        [0, 11, 19]
    """
    name = "hour"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(24))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = 'the hour value of {}'

    def get_function(self):
        def hour(vals):
            return vals.dt.hour
        return hour


class IsWeekend(TransformPrimitive):
    """Determines if a date falls on a weekend.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> is_weekend = IsWeekend()
        >>> is_weekend(dates).tolist()
        [False, False, True]
    """
    name = "is_weekend"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} occurred on a weekend"

    def get_function(self):
        def is_weekend(vals):
            return vals.dt.weekday > 4
        return is_weekend


class Minute(TransformPrimitive):
    """Determines the minutes value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> minute = Minute()
        >>> minute(dates).tolist()
        [0, 10, 45]
    """
    name = "minute"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(60))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the minutes value of {}"

    def get_function(self):
        def minute(vals):
            return vals.dt.minute
        return minute


class Month(TransformPrimitive):
    """Determines the month value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> month = Month()
        >>> month(dates).tolist()
        [3, 6, 11]
    """
    name = "month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 13))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the month of {}"

    def get_function(self):
        def month(vals):
            return vals.dt.month
        return month


class Second(TransformPrimitive):
    """Determines the seconds value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> second = Second()
        >>> second(dates).tolist()
        [0, 50, 15]
    """
    name = "second"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(60))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the seconds value of {}"

    def get_function(self):
        def second(vals):
            return vals.dt.second
        return second


class TimeSince(TransformPrimitive):
    """Calculates time from a value to a specified cutoff datetime.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to Seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> time_since = TimeSince()
        >>> times = [datetime(2019, 3, 1, 0, 0, 0, 1),
        ...          datetime(2019, 3, 1, 0, 0, 1, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0, 0)]
        >>> cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> values = time_since(times, time=cutoff_time)
        >>> list(map(int, values))
        [0, -1, -120]

        Change output to nanoseconds

        >>> from datetime import datetime
        >>> time_since_nano = TimeSince(unit='nanoseconds')
        >>> times = [datetime(2019, 3, 1, 0, 0, 0, 1),
        ...          datetime(2019, 3, 1, 0, 0, 1, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0, 0)]
        >>> cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> values = time_since_nano(times, time=cutoff_time)
        >>> list(map(lambda x: int(round(x)), values))
        [-1000, -1000000000, -120000000000]
    """
    name = 'time_since'
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the time from {} to the cutoff time"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self):
        def pd_time_since(array, time):
            return convert_time_units((time - array).dt.total_seconds(), self.unit)
        return pd_time_since


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous entry in a list.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to Seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Description:
        Given a list of datetimes, compute the time in seconds elapsed since
        the previous item in the list. The result for the first item in the
        list will always be `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_previous = TimeSincePrevious()
        >>> dates = [datetime(2019, 3, 1, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0),
        ...          datetime(2019, 3, 1, 0, 3, 0),
        ...          datetime(2019, 3, 1, 0, 2, 30),
        ...          datetime(2019, 3, 1, 0, 10, 0)]
        >>> time_since_previous(dates).tolist()
        [nan, 120.0, 60.0, -30.0, 450.0]
    """
    name = "time_since_previous"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the time since the previous instance of {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self):
        def pd_diff(values):
            return convert_time_units(values.diff().apply(lambda x: x.total_seconds()), self.unit)
        return pd_diff


class Week(TransformPrimitive):
    """Determines the week of the year from a datetime.

    Description:
        Returns the week of the year from a datetime value. The first week
        of the year starts on January 1, and week numbers increment each
        Monday.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 1, 3),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> week = Week()
        >>> week(dates).tolist()
        [1, 25, 48]
        """
    name = "week"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 54))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the week of the year of {}"

    def get_function(self):
        def week(vals):
            warnings.filterwarnings("ignore",
                                    message=("Series.dt.weekofyear and Series.dt.week "
                                             "have been deprecated."),
                                    module="featuretools"
                                    )
            return vals.dt.week
        return week


class Weekday(TransformPrimitive):
    """Determines the day of the week from a datetime.

    Description:
        Returns the day of the week from a datetime value. Weeks
        start on Monday (day 0) and run through Sunday (day 6).

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> weekday = Weekday()
        >>> weekday(dates).tolist()
        [4, 0, 5]
    """
    name = "weekday"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(7))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the day of the week of {}"

    def get_function(self):
        def weekday(vals):
            return vals.dt.weekday
        return weekday


class Year(TransformPrimitive):
    """Determines the year value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2048, 6, 17, 11, 10, 50),
        ...          datetime(1950, 11, 30, 19, 45, 15)]
        >>> year = Year()
        >>> year(dates).tolist()
        [2019, 2048, 1950]
    """
    name = "year"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 3000))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the year of {}"

    def get_function(self):
        def year(vals):
            return vals.dt.year
        return year


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

    def __init__(self, country='US'):
        self.country = country
        try:
            self.holidays = holidays.country_holidays(country=self.country)
        except NotImplementedError:
            available_countries = 'https://github.com/dr-prodigy/python-holidays#available-countries'
            error = 'must be one of the available countries:\n%s' % available_countries
            raise ValueError(error)
        years_list = [1950 + x for x in range(150)]
        self.federal_holidays = getattr(holidays, country)(years=years_list)

    def get_function(self):
        def is_federal_holiday(x):
            holidays_df = pd.DataFrame(sorted(self.federal_holidays.items()),
                                       columns=['dates', 'names'])
            is_holiday = x.dt.normalize().isin(holidays_df.dates)
            if x.isnull().values.any():
                is_holiday = is_holiday.astype('object')
                is_holiday[x.isnull()] = np.nan
            return is_holiday.values
        return is_federal_holiday
