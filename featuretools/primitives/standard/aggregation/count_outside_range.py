import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class CountOutsideRange(AggregationPrimitive):
    """Determines the number of values that fall outside a certain range.

    Args:
        lower (float): Lower boundary of range (exclusive). Default is 0.
        upper (float): Upper boundary of range (exclusive). Default is 1.
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> count_outside_range = CountOutsideRange(lower=1.5, upper=3.6)
        >>> count_outside_range([1, 2, 3, 4, 5])
        3

        The way NaNs are treated can be controlled.

        >>> count_outside_range_skipna = CountOutsideRange(skipna=False)
        >>> count_outside_range_skipna([1, 2, 3, 4, 5, None])
        nan
    """

    name = "count_outside_range"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, lower=0, upper=1, skipna=True):
        self.lower = lower
        self.upper = upper
        self.skipna = skipna

    def get_function(self):
        def count_outside_range(x):
            if not self.skipna and x.isnull().values.any():
                return np.nan
            cond = (x < self.lower) | (x > self.upper)
            return cond.sum()

        return count_outside_range
