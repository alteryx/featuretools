from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools.primitives.core.aggregation_primitive import AggregationPrimitive
from featuretools.utils.gen_utils import Library
from featuretools.utils.time_utils import (
    convert_datetime_to_floats,
    convert_timedelta_to_floats,
)


class Trend(AggregationPrimitive):
    """Calculates the trend of a column over time.

    Description:
        Given a list of values and a corresponding list of
        datetimes, calculate the slope of the linear trend
        of values.

    Examples:
        >>> from datetime import datetime
        >>> trend = Trend()
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30),
        ...          datetime(2010, 1, 1, 11, 12),
        ...          datetime(2010, 1, 1, 11, 12, 15)]
        >>> round(trend([1, 2, 3, 4, 5], times), 3)
        -0.053
    """

    name = "trend"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the linear trend of {} over time"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_trend(y, x):
            df = pd.DataFrame({"x": x, "y": y}).dropna()
            if df.shape[0] <= 2:
                return np.nan
            if isinstance(df["x"].iloc[0], (datetime, pd.Timestamp)):
                x = convert_datetime_to_floats(df["x"])
            else:
                x = df["x"].values

            if isinstance(df["y"].iloc[0], (datetime, pd.Timestamp)):
                y = convert_datetime_to_floats(df["y"])
            elif isinstance(df["y"].iloc[0], (timedelta, pd.Timedelta)):
                y = convert_timedelta_to_floats(df["y"])
            else:
                y = df["y"].values

            x = x - x.mean()
            y = y - y.mean()

            # prevent divide by zero error
            if len(np.unique(x)) == 1:
                return 0

            # consider scipy.stats.linregress for large n cases
            coefficients = np.polyfit(x, y, 1)

            return coefficients[0]

        return pd_trend
