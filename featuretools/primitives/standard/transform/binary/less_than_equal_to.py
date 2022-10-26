import numpy as np
import pandas.api.types as pdtypes
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime, Ordinal

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class LessThanEqualTo(TransformPrimitive):
    """Determines if values in one list are less than or equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is less than or equal to each
        corresponding value in Y. Equal pairs will return `True`.

    Examples:
        >>> less_than_equal_to = LessThanEqualTo()
        >>> less_than_equal_to([2, 1, 2], [1, 2, 2]).tolist()
        [False, True, True]
    """

    name = "less_than_equal_to"
    input_types = [
        [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(semantic_tags={"numeric"}),
        ],
        [ColumnSchema(logical_type=Datetime), ColumnSchema(logical_type=Datetime)],
        [ColumnSchema(logical_type=Ordinal), ColumnSchema(logical_type=Ordinal)],
    ]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is less than or equal to {}"

    def get_function(self):
        def less_than_equal(val1, val2):
            val1_is_categorical = pdtypes.is_categorical_dtype(val1)
            val2_is_categorical = pdtypes.is_categorical_dtype(val2)
            if val1_is_categorical and val2_is_categorical:
                if not all(val1.cat.categories == val2.cat.categories):
                    return np.nan
            elif val1_is_categorical or val2_is_categorical:
                # This can happen because CFM does not set proper dtypes for intermediate
                # features, so some agg features that should be Ordinal don't yet have correct type.
                return np.nan
            return val1 <= val2

        return less_than_equal

    def generate_name(self, base_feature_names):
        return "%s <= %s" % (base_feature_names[0], base_feature_names[1])
