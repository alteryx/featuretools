import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Datetime, Double

from featuretools.primitives.base import TransformPrimitive


class CumulativeTimeSinceLastFalse(TransformPrimitive):
    """Determines the time since last `False` value.

    Description:
        Given a list of booleans and a list of corresponding
        datetimes, determine the time at each point since the
        last `False` value. Returns time difference in seconds.
        `NaN` values are ignored.

    Examples:
        >>> from datetime import datetime
        >>> cumulative_time_since_last_false = CumulativeTimeSinceLastFalse()
        >>> booleans = [False, True, False, True]
        >>> datetimes = [
        ...     datetime(2011, 4, 9, 10, 30, 0),
        ...     datetime(2011, 4, 9, 10, 30, 10),
        ...     datetime(2011, 4, 9, 10, 30, 15),
        ...     datetime(2011, 4, 9, 10, 30, 29)
        ... ]
        >>> cumulative_time_since_last_false(datetimes, booleans).tolist()
        [0.0, 10.0, 0.0, 14.0]
    """

    name = "cumulative_time_since_last_false"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(logical_type=Boolean),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def get_function(self):
        def time_since_previous_false(datetime_col, bool_col):
            if bool_col.dropna().empty:
                return pd.Series([np.nan] * len(bool_col))
            df = pd.DataFrame(
                {
                    "datetime": datetime_col,
                    "last_false_datetime": datetime_col,
                    "bool": bool_col,
                },
            )
            not_false_indices = df["bool"]
            df.loc[not_false_indices, "last_false_datetime"] = np.nan
            df["last_false_datetime"] = df["last_false_datetime"].fillna(method="ffill")
            total_seconds = (
                pd.to_datetime(df["datetime"]).subtract(df["last_false_datetime"])
            ).dt.total_seconds()
            return pd.Series(total_seconds)

        return time_since_previous_false
