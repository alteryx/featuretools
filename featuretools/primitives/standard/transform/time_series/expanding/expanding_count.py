import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.standard.transform.time_series.utils import (
    _apply_gap_for_expanding_primitives,
)


class ExpandingCount(TransformPrimitive):
    """Computes the expanding count of events over a given window.

    Description:
        Given a list of datetimes, return an expanding count starting
        at the row `gap` rows away from the current row. An expanding
        primitive calculates the value of a primitive for a given time
        with all the data available up to the corresponding point in time.

        Input datetimes should be monotonic.

    Args:
        gap (int, optional): Specifies a gap backwards from each instance before the
            usable data begins. Corresponds to number of rows. Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Defaults to 1.


    Examples:
        >>> import pandas as pd
        >>> expanding_count = ExpandingCount()
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> expanding_count(times).tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0]

        We can also control the gap before the expanding calculation.

        >>> import pandas as pd
        >>> expanding_count = ExpandingCount(gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> expanding_count(times).tolist()
        [1.0, 2.0, 3.0, 4.0, 5.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> expanding_count = ExpandingCount(min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> expanding_count(times).tolist()
        [nan, nan, 2.0, 3.0, 4.0]
    """

    name = "expanding_count"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_full_dataframe = True

    def __init__(self, gap=1, min_periods=0):
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def expanding_count(datetime):
            x = pd.Series(1, index=datetime)
            x = _apply_gap_for_expanding_primitives(x, self.gap)
            return x.expanding(min_periods=self.min_periods).count().values

        return expanding_count
