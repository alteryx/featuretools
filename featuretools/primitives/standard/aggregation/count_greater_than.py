from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


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
