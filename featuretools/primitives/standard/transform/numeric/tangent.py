import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Tangent(TransformPrimitive):
    """Computes the tangent of a number.

    Examples:
        >>> tan = Tangent()
        >>> tan([-np.pi, 0.0, np.pi/2.0]).tolist()
        [1.2246467991473532e-16, 0.0, 1.633123935319537e+16]
    """

    name = "tangent"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the tangent of {}"

    def get_function(self):
        return np.tan
