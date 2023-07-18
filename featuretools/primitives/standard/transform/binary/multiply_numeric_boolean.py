import pandas.api.types as pdtypes
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class MultiplyNumericBoolean(TransformPrimitive):
    """Performs element-wise multiplication of a numeric list with a boolean list.

    Description:
        Given a list of numeric values X and a list of
        boolean values Y, return the values in X where
        the corresponding value in Y is True.

    Examples:
        >>> import pandas as pd
        >>> multiply_numeric_boolean = MultiplyNumericBoolean()
        >>> multiply_numeric_boolean([2, 1, 2], [True, True, False]).tolist()
        [2, 1, 0]
        >>> multiply_numeric_boolean([2, None, None], [True, True, False]).astype("float64").tolist()
        [2.0, nan, nan]
        >>> multiply_numeric_boolean([2, 1, 2], pd.Series([True, True, pd.NA], dtype="boolean")).tolist()
        [2, 1, <NA>]
    """

    name = "multiply_numeric_boolean"
    input_types = [
        [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(logical_type=Boolean),
        ],
        [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(logical_type=BooleanNullable),
        ],
        [
            ColumnSchema(logical_type=Boolean),
            ColumnSchema(semantic_tags={"numeric"}),
        ],
        [
            ColumnSchema(logical_type=BooleanNullable),
            ColumnSchema(semantic_tags={"numeric"}),
        ],
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK]
    commutative = True
    description_template = "the product of {} and {}"

    def get_function(self):
        def multiply_numeric_boolean(ser1, ser2):
            if pdtypes.is_bool_dtype(ser1):
                bools = ser1
                vals = ser2
            else:
                bools = ser2
                vals = ser1
            result = vals * bools.astype("Int64")
            return result

        return multiply_numeric_boolean

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], base_feature_names[1])
