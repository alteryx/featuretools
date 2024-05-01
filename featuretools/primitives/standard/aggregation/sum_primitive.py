import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.primitives.standard.aggregation.count import Count


class Sum(AggregationPrimitive):
    """Calculates the total addition, ignoring `NaN`.

    Examples:
        >>> sum = Sum()
        >>> sum([1, 2, 3, 4, 5, None])
        15.0
    """

    name = "sum"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    stack_on_exclude = [Count]
    default_value = 0
    description_template = "the sum of {}"

    def get_function(self):
        return np.sum
