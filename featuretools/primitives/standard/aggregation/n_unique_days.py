from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

from featuretools.primitives.base import AggregationPrimitive


class NUniqueDays(AggregationPrimitive):
    """Determines the number of unique days.

    Description:
        Given a list of datetimes, return the number of unique days.
        The same day in two different years is treated as different. So
        Feb 21, 2017 is different than Feb 21, 2019, even though they are
        both the 21st of February.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days = NUniqueDays()
        >>> times = [datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 1)]
        >>> n_unique_days(times)
        3
    """

    name = "n_unique_days"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def n_unique_days(x):
            return x.dt.floor("D").nunique()

        return n_unique_days
