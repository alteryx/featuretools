from pandas import NaT
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools.primitives.base import AggregationPrimitive


class DateFirstEvent(AggregationPrimitive):
    """Determines the first datetime from a list of datetimes.

    Examples:
        >>> from datetime import datetime
        >>> date_first_event = DateFirstEvent()
        >>> date_first_event([
        ...     datetime(2011, 4, 9, 10, 30, 10),
        ...     datetime(2011, 4, 9, 10, 30, 20),
        ...     datetime(2011, 4, 9, 10, 30, 30)])
        Timestamp('2011-04-09 10:30:10')
    """

    name = "date_first_event"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Datetime)
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def date_first_event(x):
            x = x.dropna()
            if x.empty:
                return NaT
            return x.iat[0]

        return date_first_event
