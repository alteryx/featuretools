import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class Any(AggregationPrimitive):
    """Determines if any value is 'True' in a list.

    Description:
        Given a list of booleans, return `True` if one or
        more of the values are `True`.

    Examples:
        >>> any = Any()
        >>> any([False, False, False, True])
        True
    """

    name = "any"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=Boolean)
    stack_on_self = False
    description_template = "whether any of {} are true"

    def get_function(self):
        return np.any
