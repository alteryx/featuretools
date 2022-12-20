from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer

from featuretools.primitives.base import AggregationPrimitive


class MaxConsecutiveZeros(AggregationPrimitive):
    """Determines the maximum number of consecutive zero values in the input

    Args:
        skipna (bool): Ignore any `NaN` values in the input. Default is True.

    Examples:
        >>> max_consecutive_zeros = MaxConsecutiveZeros()
        >>> max_consecutive_zeros([1.0, -1.4, 0, 0.0, 0, -4.3])
        3

        `NaN` values can be ignored with the `skipna` parameter

        >>> max_consecutive_zeros_skipna = MaxConsecutiveZeros(skipna=False)
        >>> max_consecutive_zeros_skipna([1.0, -1.4, 0, None, 0.0, -4.3])
        1
    """

    name = "max_consecutive_zeros"
    input_types = [
        [ColumnSchema(logical_type=Integer)],
        [ColumnSchema(logical_type=Double)],
    ]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_zeros(x):
            if self.skipna:
                x = x.dropna()
            # convert the numeric values to booleans for processing
            x[x.notnull()] = x[x.notnull()].eq(0)
            # find the locations where the value changes from the previous value
            not_equal = x != x.shift()
            # Use cumulative sum to determine where consecutive values occur. When the
            # sum changes, consecutive non-zero values are present, when the cumulative
            # sum remains unchnaged, consecutive zero values are present.
            not_equal_sum = not_equal.cumsum()
            # group the input by the cumulative sum values and use cumulative count
            # to count the number of consecutive values. Add 1 to account for the cumulative
            # sum starting at zero where the first zero occurs
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            # multiply by the boolean input to keep only the counts that correspond to
            # zero values
            consecutive_zero = consecutive * x
            # return the max of all the consecutive zero values
            return consecutive_zero.max()

        return max_consecutive_zeros
