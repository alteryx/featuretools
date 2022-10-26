from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class DivideByFeature(TransformPrimitive):
    """Divides a scalar by each value in the list.

    Description:
        Given a list of numeric values and a scalar, divide
        the scalar by each value and return the list of
        quotients.

    Examples:
        >>> divide_by_feature = DivideByFeature(value=2)
        >>> divide_by_feature([4, 1, 2]).tolist()
        [0.5, 2.0, 1.0]
    """

    name = "divide_by_feature"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=1):
        self.value = value
        self.description_template = "the result of {} divided by {{}}".format(
            self.value,
        )

    def get_function(self):
        def divide_by_feature(vals):
            return self.value / vals

        return divide_by_feature

    def generate_name(self, base_feature_names):
        return "%s / %s" % (str(self.value), base_feature_names[0])
