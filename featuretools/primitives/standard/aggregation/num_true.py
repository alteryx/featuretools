import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class NumTrue(AggregationPrimitive):
    """Counts the number of `True` values.

    Description:
        Given a list of booleans, return the number
        of `True` values. Ignores 'NaN'.

    Examples:
        >>> num_true = NumTrue()
        >>> num_true([True, False, True, True, None])
        3
    """

    name = "num_true"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0
    stack_on = []
    stack_on_exclude = []
    description_template = "the number of times {} is true"

    def get_function(self):
        return np.sum
