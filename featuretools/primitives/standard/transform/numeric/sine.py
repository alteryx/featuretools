import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Sine(TransformPrimitive):
    """Computes the sine of a number.

    Examples:
        >>> sin = Sine()
        >>> sin([-np.pi/2.0, 0.0, np.pi/2.0]).tolist()
        [-1.0, 0.0, 1.0]
    """

    name = "sine"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the sine of {}"

    def get_function(self):
        return np.sin
