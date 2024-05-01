import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class All(AggregationPrimitive):
    """Calculates if all values are 'True' in a list.

    Description:
        Given a list of booleans, return `True` if all
        of the values are `True`.

    Examples:
        >>> all = All()
        >>> all([False, False, False, True])
        False
    """

    name = "all"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=Boolean)
    stack_on_self = False
    description_template = "whether all of {} are true"

    def get_function(self):
        return np.all
