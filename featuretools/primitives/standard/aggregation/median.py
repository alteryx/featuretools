import pandas as pd
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
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
