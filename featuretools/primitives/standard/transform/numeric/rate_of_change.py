import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class RateOfChange(TransformPrimitive):
    """Computes the rate of change of a value per second.

    Examples:
        >>> import pandas as pd
        >>> rate_of_change = RateOfChange()
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rate_of_change(times, [0, 1, 3, -1, 0]).tolist()
        [nan, 4.0, 3.0, 2.0, 1.0]
    """

    name = "rate_of_change"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"time_index"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_full_dataframe = True
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the rate of change of of {}"

    def get_function(self):
        def rate_of_change(values, time):
            return 1

        return rate_of_change
