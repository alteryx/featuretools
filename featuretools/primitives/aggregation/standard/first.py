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


class First(AggregationPrimitive):
    """Determines the first value in a list.

    Examples:
        >>> first = First()
        >>> first([1, 2, 3, 4, 5, None])
        1.0
    """

    name = "first"
    input_types = [ColumnSchema()]
    return_type = None
    stack_on_self = False
    description_template = "the first instance of {}"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_first(x):
            return x.iloc[0]

        return pd_first
