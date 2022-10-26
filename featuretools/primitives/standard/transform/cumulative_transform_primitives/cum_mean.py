import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class CumMean(TransformPrimitive):
    """Calculates the cumulative mean.

    Description:
        Given a list of values, return the cumulative mean
        (or running mean). There is no set window, so the
        mean at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're treated as 0.

    Examples:
        >>> cum_mean = CumMean()
        >>> cum_mean([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 1.5, 2.0, 2.5, nan, 2.5]
    """

    name = "cum_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    description_template = "the cumulative mean of {}"

    def get_function(self):
        def cum_mean(values):
            return values.cumsum() / np.arange(1, len(values) + 1)

        return cum_mean
