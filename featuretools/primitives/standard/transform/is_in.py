from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsIn(TransformPrimitive):
    """Determines whether a value is present in a provided list.

    Examples:
        >>> items = ['string', 10.3, False]
        >>> is_in = IsIn(list_of_outputs=items)
        >>> is_in(['string', 10.5, False]).tolist()
        [True, False, True]
    """

    name = "isin"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs
        if not list_of_outputs:
            stringified_output_list = "[]"
        else:
            stringified_output_list = ", ".join([str(x) for x in list_of_outputs])
        self.description_template = "whether {{}} is in {}".format(
            stringified_output_list,
        )

    def get_function(self):
        def pd_is_in(array):
            return array.isin(self.list_of_outputs or [])

        return pd_is_in

    def generate_name(self, base_feature_names):
        return "%s.isin(%s)" % (base_feature_names[0], str(self.list_of_outputs))
