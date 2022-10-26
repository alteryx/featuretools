import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Count(AggregationPrimitive):
    """Determines the total number of values, excluding `NaN`.

    Examples:
        >>> count = Count()
        >>> count([1, 2, 3, 4, 5, None])
        5
    """

    name = "count"
    input_types = [ColumnSchema(semantic_tags={"index"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "count"

        return pd.Series.count

    def generate_name(
        self,
        base_feature_names,
        relationship_path_name,
        parent_dataframe_name,
        where_str,
        use_prev_str,
    ):
        return "COUNT(%s%s%s)" % (relationship_path_name, where_str, use_prev_str)
