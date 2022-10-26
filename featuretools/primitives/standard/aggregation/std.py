import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Std(AggregationPrimitive):
    """Computes the dispersion relative to the mean value, ignoring `NaN`.

    Examples:
        >>> std = Std()
        >>> round(std([1, 2, 3, 4, 5, None]), 3)
        1.414
    """

    name = "std"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the standard deviation of {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "std"

        return np.std
