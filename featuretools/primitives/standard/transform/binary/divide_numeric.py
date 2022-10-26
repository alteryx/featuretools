from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class DivideNumeric(TransformPrimitive):
    """Performs element-wise division of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the quotient of each value in X
        divided by its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x / y and y / x, or just one. If True, there is
            no guarantee which of the two will be generated. Defaults to False.

    Examples:
        >>> divide_numeric = DivideNumeric()
        >>> divide_numeric([2.0, 1.0, 2.0], [1.0, 2.0, 2.0]).tolist()
        [2.0, 0.5, 1.0]
    """

    name = "divide_numeric"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the result of {} divided by {}"

    def __init__(self, commutative=False):
        self.commutative = commutative

    def get_function(self):
        def divide_numeric(val1, val2):
            return val1 / val2

        return divide_numeric

    def generate_name(self, base_feature_names):
        return "%s / %s" % (base_feature_names[0], base_feature_names[1])
