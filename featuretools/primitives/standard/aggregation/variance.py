import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import AggregationPrimitive


class Variance(AggregationPrimitive):
    """Calculates the variance of a list of numbers.

    Description:
        Given a list of numbers, return the variance,
        using numpy's built-in variance function. Nan
        values in a series will be ignored. Return nan
        when the series is empty or entirely null.

    Examples:
        >>> variance = Variance()
        >>> variance([0, 3, 4, 3])
        2.25

        Null values in a series will be ignored.

        >>> variance = Variance()
        >>> variance([0, 3, 4, 3, None])
        2.25
    """

    name = "variance"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = np.nan

    def get_function(self):
        return np.var
