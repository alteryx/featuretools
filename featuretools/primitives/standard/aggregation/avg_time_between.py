from datetime import datetime

import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library


class AvgTimeBetween(AggregationPrimitive):
    """Computes the average number of seconds between consecutive events.

    Description:
        Given a list of datetimes, return the average time (default in seconds)
        elapsed between consecutive events. If there are fewer
        than 2 non-null values, return `NaN`.

    Args:
        unit (str): Defines the unit of time.
            Defaults to seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> avg_time_between = AvgTimeBetween()
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> avg_time_between(times)
        375.0
        >>> avg_time_between = AvgTimeBetween(unit="minutes")
        >>> avg_time_between(times)
        6.25
    """

    name = "avg_time_between"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    description_template = "the average time between each of {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self, agg_type=Library.PANDAS):
        def pd_avg_time_between(x):
            """Assumes time scales are closer to order
            of seconds than to nanoseconds
            if times are much closer to nanoseconds
            we could get some floating point errors

            this can be fixed with another function
            that calculates the mean before converting
            to seconds
            """
            x = x.dropna()
            if x.shape[0] < 2:
                return np.nan
            if isinstance(x.iloc[0], (pd.Timestamp, datetime)):
                x = x.view("int64")
                # use len(x)-1 because we care about difference
                # between values, len(x)-1 = len(diff(x))

            avg = (x.max() - x.min()) / (len(x) - 1)
            avg = avg * 1e-9

            # long form:
            # diff_in_ns = x.diff().iloc[1:].astype('int64')
            # diff_in_seconds = diff_in_ns * 1e-9
            # avg = diff_in_seconds.mean()
            return convert_time_units(avg, self.unit)

        return pd_avg_time_between
