import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number.

    Examples:
        >>> absolute = Absolute()
        >>> absolute([3.0, -5.0, -2.4]).tolist()
        [3.0, 5.0, 2.4]
    """

    name = "absolute"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    description_template = "the absolute value of {}"

    def get_function(self):
        return np.absolute
