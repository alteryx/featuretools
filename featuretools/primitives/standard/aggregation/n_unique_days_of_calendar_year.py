from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

from featuretools.primitives.base import AggregationPrimitive


class NUniqueDaysOfCalendarYear(AggregationPrimitive):
    """Determines the number of unique calendar days.

    Description:
        Given a list of datetimes, return the number of unique calendar
        days. The same date in two different years is counted as one. So
        Feb 21, 2017 is not unique from Feb 21, 2019.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days_of_calendar_year = NUniqueDaysOfCalendarYear()
        >>> times = [datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 1)]
        >>> n_unique_days_of_calendar_year(times)
        2
    """

    name = "n_unique_days_of_calendar_year"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def n_unique_days_of_calendar_year(x):
            return x.dropna().dt.strftime("%m-%d").nunique()

        return n_unique_days_of_calendar_year
