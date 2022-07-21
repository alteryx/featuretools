import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Datetime, Ordinal

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.utils.gen_utils import Library


class LessThanScalar(TransformPrimitive):
    """Determines if values are less than a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is less than the scalar.
        If a value is equal to the scalar, return `False`.

    Examples:
        >>> less_than_scalar = LessThanScalar(value=2)
        >>> less_than_scalar([3, 1, 2]).tolist()
        [False, True, False]
    """

    name = "less_than_scalar"
    input_types = [
        [ColumnSchema(semantic_tags={"numeric"})],
        [ColumnSchema(logical_type=Datetime)],
        [ColumnSchema(logical_type=Ordinal)],
    ]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=0):
        self.value = value
        self.description_template = "whether {{}} is less than {}".format(self.value)

    def get_function(self):
        def less_than_scalar(vals):
            if (
                pdtypes.is_categorical_dtype(vals)
                and self.value not in vals.cat.categories
            ):
                return np.nan
            return vals < self.value

        return less_than_scalar

    def generate_name(self, base_feature_names):
        return "%s < %s" % (base_feature_names[0], str(self.value))
