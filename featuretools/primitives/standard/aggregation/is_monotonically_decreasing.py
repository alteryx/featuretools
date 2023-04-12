from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base import AggregationPrimitive


class IsMonotonicallyDecreasing(AggregationPrimitive):
    """Determines if a series is monotonically decreasing.

    Description:
        Given a list of numeric values, return True if the
        values are strictly decreasing. If the series contains
        `NaN` values, they will be skipped.

    Examples:
        >>> is_monotonically_decreasing = IsMonotonicallyDecreasing()
        >>> is_monotonically_decreasing([9, 5, 3, 1])
        True
    """

    name = "is_monotonically_decreasing"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    stack_on_self = False
    default_value = False

    def get_function(self):
        def is_monotonically_decreasing(x):
            return x.dropna().is_monotonic_decreasing

        return is_monotonically_decreasing
