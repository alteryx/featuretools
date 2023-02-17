from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import AggregationPrimitive


class PercentUnique(AggregationPrimitive):
    """Determines the percent of unique values.

    Description:
        Given a list of values, determine what percent of the
        list is made up of unique values.  Multiple `NaN` values
        are treated as one unique value.

    Args:
        skipna (bool): Determines whether to ignore `NaN` values.
            Defaults to True.

    Examples:
        >>> percent_unique = PercentUnique()
        >>> percent_unique([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
        0.8

        We can control whether or not `NaN` values are ignored.

        >>> percent_unique = PercentUnique()
        >>> percent_unique([1, 1, 2, None])
        0.5
        >>> percent_unique_skipna = PercentUnique(skipna=False)
        >>> percent_unique_skipna([1, 1, 2, None])
        0.75
    """

    name = "percent_unique"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def percent_unique(x):
            return x.nunique(dropna=self.skipna) / (x.shape[0] * 1.0)

        return percent_unique
