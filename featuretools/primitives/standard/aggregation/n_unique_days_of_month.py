from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

from featuretools.primitives.base import AggregationPrimitive


class NUniqueDaysOfMonth(AggregationPrimitive):
    """Determines the number of unique days of month.

    Description:
        Given a list of datetimes, return the number of unique days
        of month. The maximum value is 31. 2018-01-01 and 2018-02-01
        will be counted as 1 unique day. 2019-01-01 and 2018-01-01
        will also be counted as 1.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days_of_month = NUniqueDaysOfMonth()
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 2),
        ...          datetime(2019, 1, 3)]
        >>> n_unique_days_of_month(times)
        3
    """

    name = "n_unique_days_of_month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def n_unique_days_of_month(x):
            return x.dropna().dt.day.nunique()

        return n_unique_days_of_month
