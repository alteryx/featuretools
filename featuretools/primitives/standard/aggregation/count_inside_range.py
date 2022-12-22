import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class CountInsideRange(AggregationPrimitive):
    """Determines the number of values that fall within a certain range.

    Args:
        lower (float): Lower boundary of range (inclusive). Default is 0.
        upper (float): Upper boundary of range (inclusive). Default is 1.
        skipna (bool): If this is False any value in x is NaN then
            the result will be NaN. If True, `nan` values are skipped.
            Default is True.

    Examples:
        >>> count_inside_range = CountInsideRange(lower=1.5, upper=3.6)
        >>> count_inside_range([1, 2, 3, 4, 5])
        2

        The way NaNs are treated can be controlled.

        >>> count_inside_range_skipna = CountInsideRange(skipna=False)
        >>> count_inside_range_skipna([1, 2, 3, 4, 5, None])
        nan
    """

    name = "count_inside_range"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, lower=0, upper=1, skipna=True):
        self.lower = lower
        self.upper = upper
        self.skipna = skipna

    def get_function(self):
        def count_inside_range(x):
            if not self.skipna and x.isnull().values.any():
                return np.nan
            cond = (self.lower <= x) & (x <= self.upper)
            return cond.sum()

        return count_inside_range
