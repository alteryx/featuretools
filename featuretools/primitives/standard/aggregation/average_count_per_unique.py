from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import AggregationPrimitive


class AverageCountPerUnique(AggregationPrimitive):
    """Determines the average count across all unique value.

    Args:
        skipna (bool): Determines if to use NA/null values.
            Defaults to True to skip NA/null.

    Examples:
        Determine the average count values for all unique items
        in the input
        >>> input = [1, 1, 2, 2, 3, 4, 5, 6, 7, 8]
        >>> avg_count_per_unique = AverageCountPerUnique()
        >>> avg_count_per_unique(input)
        1.25

        Determine the average count values for all unique items
        in the input with nan values ignored
        >>> input = [1, 1, 2, 2, 3, 4, 5, None, 6, 7, 8]
        >>> avg_count_per_unique = AverageCountPerUnique()
        >>> avg_count_per_unique(input)
        1.25

        Determine the average count values for all unique items
        in the input with nan values included
        >>> input = [1, 2, 2, 3, 4, 5, None, 6, 7, 8, 9]
        >>> avg_count_per_unique_skipna_false = AverageCountPerUnique(skipna=False)
        >>> avg_count_per_unique_skipna_false(input)
        1.1
    """

    name = "average_count_per_unique"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def average_count_per_unique(x):
            return x.value_counts(
                dropna=self.skipna,
            ).mean(skipna=self.skipna)

        return average_count_per_unique
