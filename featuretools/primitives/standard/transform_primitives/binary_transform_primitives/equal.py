import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Equal(TransformPrimitive):
    """Determines if values in one list are equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is equal to each corresponding value
        in Y.

    Examples:
        >>> equal = Equal()
        >>> equal([2, 1, 2], [1, 2, 2]).tolist()
        [False, False, True]
    """

    name = "equal"
    input_types = [ColumnSchema(), ColumnSchema()]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    commutative = True
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} equals {}"

    def get_function(self):
        def equal(x_vals, y_vals):
            if isinstance(x_vals.dtype, pd.CategoricalDtype) and isinstance(
                y_vals.dtype,
                pd.CategoricalDtype,
            ):
                categories = set(x_vals.cat.categories).union(
                    set(y_vals.cat.categories),
                )
                x_vals = x_vals.cat.add_categories(
                    categories.difference(set(x_vals.cat.categories)),
                )
                y_vals = y_vals.cat.add_categories(
                    categories.difference(set(y_vals.cat.categories)),
                )
            return x_vals.eq(y_vals)

        return equal

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], base_feature_names[1])
