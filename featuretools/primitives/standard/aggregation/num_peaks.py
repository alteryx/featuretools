import pandas as pd
from scipy.signal import find_peaks
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer

from featuretools.primitives.base import AggregationPrimitive


class NumPeaks(AggregationPrimitive):
    """Determines the number of peaks in a list of numbers.

    Description:
        Given a list of numbers, count the number of local
        maxima. Uses the find_peaks function from scipy.signal.

    Examples:
        >>> num_peaks = NumPeaks()
        >>> num_peaks([-5, 0, 10, 0, 10, -5, -4, -5, 10, 0])
        4
    """

    name = "num_peaks"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def num_peaks(x):
            if x.dtype == "Int64":
                x = x.astype("float64")
            peaks = find_peaks(x)[0]
            return len(peaks[~pd.isna(peaks)])

        return num_peaks
