import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class ModuloNumeric(TransformPrimitive):
    """Performs element-wise modulo of two lists.

    Description:
        Given a list of values X and a list of values Y,
        determine the modulo, or remainder of each value in
        X after it's divided by its corresponding value in Y.

    Examples:
        >>> modulo_numeric = ModuloNumeric()
        >>> modulo_numeric([2, 1, 5], [1, 2, 2]).tolist()
        [0, 1, 1]
    """

    name = "modulo_numeric"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the remainder after dividing {} by {}"

    def get_function(self):
        return np.mod

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (base_feature_names[0], base_feature_names[1])
