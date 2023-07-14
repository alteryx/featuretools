from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base import AggregationPrimitive


class IsMonotonicallyIncreasing(AggregationPrimitive):
    """Determines if a series is monotonically increasing.

    Description:
        Given a list of numeric values, return True if the
        values are strictly increasing. If the series contains
        `NaN` values, they will be skipped.

    Examples:
        >>> is_monotonically_increasing = IsMonotonicallyIncreasing()
        >>> is_monotonically_increasing([1, 3, 5, 9])
        True
    """

    name = "is_monotonically_increasing"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    stack_on_self = False
    default_value = False

    def get_function(self):
        def is_monotonically_increasing(x):
            return x.dropna().is_monotonic_increasing

        return is_monotonically_increasing
