from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Negate(TransformPrimitive):
    """Negates a numeric value.

    Examples:
        >>> negate = Negate()
        >>> negate([1.0, 23.2, -7.0]).tolist()
        [-1.0, -23.2, 7.0]
    """

    name = "negate"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the negation of {}"

    def get_function(self):
        def negate(vals):
            return vals * -1

        return negate

    def generate_name(self, base_feature_names):
        return "-(%s)" % (base_feature_names[0])
