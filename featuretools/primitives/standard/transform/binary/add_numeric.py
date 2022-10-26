import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class AddNumeric(TransformPrimitive):
    """Performs element-wise addition of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the sum of each value in X with its
        corresponding value in Y.

    Examples:
        >>> add_numeric = AddNumeric()
        >>> add_numeric([2, 1, 2], [1, 2, 2]).tolist()
        [3, 3, 4]
    """

    name = "add_numeric"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    commutative = True
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the sum of {} and {}"

    def get_function(self):
        return np.add

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], base_feature_names[1])
