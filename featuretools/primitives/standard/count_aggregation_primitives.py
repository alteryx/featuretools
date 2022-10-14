import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer, IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class CountAboveMean(AggregationPrimitive):
    """Calculates the number of values that are above the mean.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> count_above_mean = CountAboveMean()
        >>> count_above_mean([1, 2, 3, 4, 5])
        2

        The way `NaN`s are treated can be controlled.

        >>> count_above_mean_skipna = CountAboveMean(skipna=False)
        >>> count_above_mean_skipna([1, 2, 3, 4, 5, None])
        nan
    """

    name = "count_above_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def count_around_mean(x):
            mean = x.mean(skipna=self.skipna)
            if np.isnan(mean):
                return np.nan
            return len(x[x > mean])

        return count_around_mean


class CountBelowMean(AggregationPrimitive):
    """Determines the number of values that are below the mean.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> count_below_mean = CountBelowMean()
        >>> count_below_mean([1, 2, 3, 4, 10])
        3

        The way `NaN`s are treated can be controlled.

        >>> count_below_mean_skipna = CountBelowMean(skipna=False)
        >>> count_below_mean_skipna([1, 2, 3, 4, 5, None])
        nan
    """

    name = "count_below_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def count_around_mean(x):
            mean = x.mean(skipna=self.skipna)
            if np.isnan(mean):
                return np.nan
            return len(x[x < mean])

        return count_around_mean


class CountGreaterThan(AggregationPrimitive):
    """Determines the number of values greater than a controllable threshold.

    Args:
        threshold (float): The threshold to use when counting the number
            of values greater than. Defaults to 10.

    Examples:
        >>> count_greater_than = CountGreaterThan(threshold=3)
        >>> count_greater_than([1, 2, 3, 4, 5])
        2
    """

    name = "count_greater_than"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, threshold=10):
        self.threshold = threshold

    def get_function(self):
        def count_greater_than(x):
            return x[x > self.threshold].count()

        return count_greater_than


class CountInsideNthSTD(AggregationPrimitive):
    """Determines the count of observations that lie inside
        the first N standard deviations (inclusive).

    Args:
        n (float): Number of standard deviations. Default is 1

    Examples:
        >>> count_inside_nth_std = CountInsideNthSTD(n=1.5)
        >>> count_inside_nth_std([1, 10, 15, 20, 100])
        4
    """

    name = "count_inside_nth_std"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, n=1):
        if n < 0:
            raise ValueError("n must be a positive number")

        self.n = n

    def get_function(self):
        def count_inside_nth_std(x):
            cond = np.abs(x - np.mean(x)) <= np.std(x) * self.n
            return cond.sum()

        return count_inside_nth_std


class CountInsideRange(AggregationPrimitive):
    """Determines the number of values that fall within a certain range.

    Args:
        lower (float): Lower boundary of range (inclusive). Default is 0.
        upper (float): Upper boundary of range (inclusive). Default is 1.
        skipna (bool): If this is False any value in x is `nan` then
            the result will be `nan`. If True, `nan` values are skipped.
            Default is True.

    Examples:
        >>> count_inside_range = CountInsideRange(lower=1.5, upper=3.6)
        >>> count_inside_range([1, 2, 3, 4, 5])
        2

        The way `NaN`s are treated can be controlled.

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

        The way `NaN`s are treated can be controlled.

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
