import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Datetime, Double

from featuretools.primitives.base import AggregationPrimitive


class TimeSinceLastFalse(AggregationPrimitive):
    """Calculates the time since the last `False` value.

    Description:
        Using a series of Datetimes and a series of Booleans, find the last
        record with a `False` value. Return the seconds elapsed between that record
        and the instance's cutoff time. Return nan if no values are `False`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_last_false = TimeSinceLastFalse()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> booleans = [True, False, True]
        >>> time_since_last_false(times, booleans, time=cutoff_time)
        285.0
    """

    name = "time_since_last_false"
    input_types = [
        [
            ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
            ColumnSchema(logical_type=Boolean),
        ],
        [
            ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
            ColumnSchema(logical_type=BooleanNullable),
        ],
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = True
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def time_since_last_false(datetime_col, bool_col, time=None):
            df = pd.DataFrame(
                {
                    "datetime": datetime_col,
                    "bool": bool_col,
                },
            ).dropna()
            if df.empty:
                return np.nan
            false_indices = df[~df["bool"]]
            if false_indices.empty:
                return np.nan
            last_false_index = false_indices.index[-1]
            time_since = time - datetime_col.loc[last_false_index]
            return time_since.total_seconds()

        return time_since_last_false
