import pandas as pd
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Skew(AggregationPrimitive):
    """Computes the extent to which a distribution differs from a normal distribution.

    Description:
        For normally distributed data, the skewness should be about 0.
        A skewness value > 0 means that there is more weight in the
        left tail of the distribution.

    Examples:
        >>> skew = Skew()
        >>> skew([1, 10, 30, None])
        1.0437603722639681
    """

    name = "skew"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on = []
    stack_on_self = False
    description_template = "the skewness of {}"

    def get_function(self, agg_type=Library.PANDAS):
        return pd.Series.skew
