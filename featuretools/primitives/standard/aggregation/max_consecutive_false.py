from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Integer

from featuretools.primitives.base import AggregationPrimitive


class MaxConsecutiveFalse(AggregationPrimitive):
    """Determines the maximum number of consecutive False values in the input

    Examples:
        >>> max_consecutive_false = MaxConsecutiveFalse()
        >>> max_consecutive_false([True, False, False, True, True, False])
        2
    """

    name = "max_consecutive_false"
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def max_consecutive_false(x):
            # invert the input array to work properly with the computation
            x[x.notnull()] = ~(x[x.notnull()].astype(bool))
            # find the locations where the value changes from the previous value
            not_equal = x != x.shift()
            # Use cumulative sum to determine where consecutive values occur. When the
            # sum changes, consecutive False values are present, when the cumulative
            # sum remains unchnaged, consecutive True values are present.
            not_equal_sum = not_equal.cumsum()
            # group the input by the cumulative sum values and use cumulative count
            # to count the number of consecutive values. Add 1 to account for the cumulative
            # sum starting at zero where the first True occurs
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            # multiply by the inverted input to keep only the counts that correspond to
            # false values
            consecutive_false = consecutive * x
            # return the max of all the consecutive false values
            return consecutive_false.max()

        return max_consecutive_false
