import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class CountOutsideNthSTD(AggregationPrimitive):
    """Determines the number of observations that lie outside
        the first N standard deviations.

    Args:
        n (float): Number of standard deviations. Default is 1

    Examples:
        >>> count_outside_nth_std = CountOutsideNthSTD(n=1.5)
        >>> count_outside_nth_std([1, 10, 15, 20, 100])
        1
    """

    name = "count_outside_nth_std"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, n=1):
        if n < 0:
            raise ValueError("n must be a positive number")

        self.n = n

    def get_function(self):
        def count_outside_nth_std(x):
            cond = np.abs(x - np.mean(x)) > np.std(x) * self.n
            return cond.sum()

        return count_outside_nth_std
