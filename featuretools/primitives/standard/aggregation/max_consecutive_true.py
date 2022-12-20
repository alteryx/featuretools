from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Integer

from featuretools.primitives.base import AggregationPrimitive


class MaxConsecutiveTrue(AggregationPrimitive):
    """Determines the maximum number of consecutive True values in the input

    Examples:
        >>> max_consecutive_true = MaxConsecutiveTrue()
        >>> max_consecutive_true([True, False, True, True, True, False])
        3
    """

    name = "max_consecutive_true"
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def max_consecutive_true(x):
            # find the locations where the value changes from the previous value
            not_equal = x != x.shift()
            # use cumulative sum to determine where consecutive values occur. When the
            # sum changes, consecutive False values are present, when the cumulative
            # sum remains unchnaged, consecutive True values are present.
            not_equal_sum = not_equal.cumsum()
            # group the input by the cumulative sum values and use cumulative count
            # to count the number of consecutive values. Add 1 to account for the cumulative
            # sum starting at zero where the first True occurs
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            # multiply by the original input to keep only the counts that correspond to
            # true values
            consecutive_true = consecutive * x
            # return the max of all the consecutive true values
            return consecutive_true.max()

        return max_consecutive_true
