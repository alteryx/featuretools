import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base import AggregationPrimitive


class FirstLastTimeDelta(AggregationPrimitive):
    """Determines the time between the first and last time value
        in seconds.

    Examples:
        >>> from datetime import datetime
        >>> first_last_time_delta = FirstLastTimeDelta()
        >>> first_last_time_delta([
        ...     datetime(2011, 4, 9, 10, 30, 0),
        ...     datetime(2011, 4, 9, 10, 30, 15),
        ...     datetime(2011, 4, 9, 10, 30, 35)])
        35.0
    """

    name = "first_last_time_delta"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = False
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def first_last_time_delta(datetime_col):
            datetime_col = datetime_col.dropna()
            if datetime_col.empty:
                return np.nan
            delta = datetime_col.iloc[-1] - datetime_col.iloc[0]
            return delta.total_seconds()

        return first_last_time_delta
