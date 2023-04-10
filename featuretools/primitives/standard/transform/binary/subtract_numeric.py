import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class SubtractNumeric(TransformPrimitive):
    """Performs element-wise subtraction of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the difference of each value
        in X from its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x - y and y - x, or just one. If True, there is no
            guarantee which of the two will be generated. Defaults to True.

    Notes:
        commutative is True by default since False would result in 2 perfectly
        correlated series.

    Examples:
        >>> subtract_numeric = SubtractNumeric()
        >>> subtract_numeric([2, 1, 2], [1, 2, 2]).tolist()
        [1, -1, 0]
    """

    name = "subtract_numeric"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the result of {} minus {}"
    commutative = True

    def __init__(self, commutative=True):
        self.commutative = commutative

    def get_function(self):
        return np.subtract

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], base_feature_names[1])
