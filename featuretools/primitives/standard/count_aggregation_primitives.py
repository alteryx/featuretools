import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer, IntegerNullable

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


class CountLessThan(AggregationPrimitive):
    """Determines the number of values less than a controllable threshold.

    Args:
        threshold (float): The threshold to use when counting the number
            of values less than. Defaults to 10.

    Examples:
        >>> count_less_than = CountLessThan(threshold=3.5)
        >>> count_less_than([1, 2, 3, 4, 5])
        3
    """

    name = "count_less_than"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, threshold=10):
        self.threshold = threshold

    def get_function(self):
        def count_less_than(x):
            return x[x < self.threshold].count()

        return count_less_than


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
