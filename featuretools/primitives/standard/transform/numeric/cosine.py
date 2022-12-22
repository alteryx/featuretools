import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Cosine(TransformPrimitive):
    """Computes the cosine of a number.

    Examples:
        >>> cos = Cosine()
        >>> cos([0.0, np.pi/2.0, np.pi]).tolist()
        [1.0, 6.123233995736766e-17, -1.0]
    """

    name = "cosine"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the cosine of {}"

    def get_function(self):
        return np.cos
