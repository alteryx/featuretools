from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base import AggregationPrimitive


class HasNoDuplicates(AggregationPrimitive):
    """Determines if there are duplicates in the input.

    Args:
        skipna (bool): Determines if to use NA/null values.
            Defaults to True to skip NA/null.

    Examples:
        >>> has_no_duplicates = HasNoDuplicates()
        >>> has_no_duplicates([1, 1, 2])
        False
        >>> has_no_duplicates([1, 2, 3])
        True

        `NaN`s are skipped by default.

        >>> has_no_duplicates([1, 2, 3, None, None])
        True

        However, the way `NaN`s are treated can be controlled.

        >>> has_no_duplicates_skipna = HasNoDuplicates(skipna=False)
        >>> has_no_duplicates_skipna([1, 2, 3, None, None])
        False
        >>> has_no_duplicates_skipna([1, 2, 3, None])
        True
    """

    name = "has_no_duplicates"
    input_types = [
        [ColumnSchema(semantic_tags={"category"})],
        [ColumnSchema(semantic_tags={"numeric"})],
    ]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    stack_on_self = False
    default_value = True

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def has_no_duplicates(data):
            if self.skipna:
                data = data.dropna()
            return not data.duplicated().any()

        return has_no_duplicates
