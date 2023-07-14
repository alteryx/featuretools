from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

from featuretools.primitives.base import AggregationPrimitive


class NUniqueMonths(AggregationPrimitive):
    """Determines the number of unique months.

    Description:
        Given a list of datetimes, return the number of unique months.
        NUniqueMonths counts absolute month, not month of year, so the
        same month in two different years is treated as different. (i.e.
        Feb 2017 is different than Feb 2019.)

    Examples:
        >>> from datetime import datetime
        >>> n_unique_months = NUniqueMonths()
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 1, 2),
        ...          datetime(2019, 1, 3),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1)]
        >>> n_unique_months(times)
        3
    """

    name = "n_unique_months"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def n_unique_months(x):
            return x.dt.to_period("M").nunique()

        return n_unique_months
