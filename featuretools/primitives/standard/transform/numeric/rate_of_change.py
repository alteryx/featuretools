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
        >>> rate_of_change([0, 30, 180, -90, 0], times).tolist()
        [nan, 0.5, 2.5, -4.5, 1.5]
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
            time_delta = time.diff().dt.total_seconds()
            value_delta = values.diff()
            return value_delta / time_delta

        return rate_of_change
