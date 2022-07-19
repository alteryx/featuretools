import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Datetime, Ordinal

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.utils.gen_utils import Library

class SubtractNumericScalar(TransformPrimitive):
    """Subtract a scalar from each element in the list.

    Description:
        Given a list of numeric values and a scalar, subtract
        the given scalar from each value in the list.

    Examples:
        >>> subtract_numeric_scalar = SubtractNumericScalar(value=2)
        >>> subtract_numeric_scalar([3, 1, 2]).tolist()
        [1, -1, 0]
    """

    name = "subtract_numeric_scalar"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=0):
        self.value = value
        self.description_template = "the result of {{}} minus {}".format(self.value)

    def get_function(self):
        def subtract_scalar(vals):
            return vals - self.value

        return subtract_scalar

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], str(self.value))