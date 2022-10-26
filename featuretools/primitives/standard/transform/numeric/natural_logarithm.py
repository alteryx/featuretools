import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class NaturalLogarithm(TransformPrimitive):
    """Computes the natural logarithm of a number.

    Examples:
        >>> log = NaturalLogarithm()
        >>> results = log([1.0, np.e]).tolist()
        >>> results = [round(x, 2) for x in results]
        >>> results
        [0.0, 1.0]
    """

    name = "natural_logarithm"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the natural logarithm of {}"

    def get_function(self):
        return np.log
