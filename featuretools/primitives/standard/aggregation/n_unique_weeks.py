from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

from featuretools.primitives.base import AggregationPrimitive


class NUniqueWeeks(AggregationPrimitive):
    """Determines the number of unique weeks.

    Description:
        Given a list of datetimes, return the number of unique
        weeks (Monday-Sunday). NUniqueWeeks counts by absolute
        week, not week of year, so the first week of 2018 and
        the first week of 2019 count as two unique values.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_weeks = NUniqueWeeks()
        >>> times = [datetime(2018, 2, 2),
        ...          datetime(2019, 1, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2019, 2, 3),
        ...          datetime(2019, 2, 21)]
        >>> n_unique_weeks(times)
        4
    """

    name = "n_unique_weeks"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def n_unique_weeks(x):
            return x.dt.to_period("W").nunique()

        return n_unique_weeks
