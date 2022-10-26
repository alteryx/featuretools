import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class SquareRoot(TransformPrimitive):
    """Computes the square root of a number.

    Examples:
        >>> sqrt = SquareRoot()
        >>> sqrt([9.0, 16.0, 4.0]).tolist()
        [3.0, 4.0, 2.0]
    """

    name = "square_root"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the square root of {}"

    def get_function(self):
        return np.sqrt
