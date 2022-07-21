from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dask import dataframe as dd
from scipy import stats
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Datetime,
    Double,
    IntegerNullable,
)

from featuretools.primitives.core.aggregation_primitive import AggregationPrimitive
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library


class Median(AggregationPrimitive):
    """Determines the middlemost number in a list of values.

    Examples:
        >>> median = Median()
        >>> median([5, 3, 2, 1, 4])
        3.0

        `NaN` values are ignored.

        >>> median([5, 3, 2, 1, 4, None])
        3.0
    """

    name = "median"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the median of {}"

    def get_function(self, agg_type=Library.PANDAS):
        return pd.Series.median
