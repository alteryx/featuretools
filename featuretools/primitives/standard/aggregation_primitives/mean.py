import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Mean(AggregationPrimitive):
    """Computes the average for a list of values.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> mean = Mean()
        >>> mean([1, 2, 3, 4, 5, None])
        3.0

        We can also control the way `NaN` values are handled.

        >>> mean = Mean(skipna=False)
        >>> mean([1, 2, 3, 4, 5, None])
        nan
    """

    name = "mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the average of {}"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "mean"

        if self.skipna:
            # np.mean of series is functionally nanmean
            return np.mean

        def mean(series):
            return np.mean(series.values)

        return mean
