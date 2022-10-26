from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsNull(TransformPrimitive):
    """Determines if a value is null.

    Examples:
        >>> is_null = IsNull()
        >>> is_null([1, None, 3]).tolist()
        [False, True, False]
    """

    name = "is_null"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is null"

    def get_function(self):
        def isnull(array):
            return array.isnull()

        return isnull
