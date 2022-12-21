import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base import AggregationPrimitive


class TimeSinceLastMax(AggregationPrimitive):
    """Calculates the time since the maximum value occurred.

    Description:
        Given a list of numbers, and a corresponding index of
        datetimes, find the time of the maximum value, and return
        the time elapsed since it occured. This calculation is done
        using an instance id's cutoff time.

        If multiple values equal the maximum, use the first occuring
        maximum.

    Examples:
        >>> from datetime import datetime
        >>> time_since_last_max = TimeSinceLastMax()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last_max(times, [1, 3, 2], time=cutoff_time)
        285.0
    """

    name = "time_since_last_max"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = True
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def time_since_last_max(datetime_col, numeric_col, time=None):
            df = pd.DataFrame(
                {
                    "datetime": datetime_col,
                    "numeric": numeric_col,
                },
            ).dropna()
            if df.empty:
                return np.nan
            max_row = df.loc[df["numeric"].idxmax()]
            max_time = max_row["datetime"]
            time_since = time - max_time
            return time_since.total_seconds()

        return time_since_last_max
