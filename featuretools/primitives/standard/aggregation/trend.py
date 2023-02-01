import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils import calculate_trend
from featuretools.utils.gen_utils import Library


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
            return calculate_trend(pd.Series(data=y.values, index=x.values))

        return pd_trend
