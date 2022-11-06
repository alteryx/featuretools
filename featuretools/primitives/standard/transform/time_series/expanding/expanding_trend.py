import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils import calculate_trend


class ExpandingTrend(TransformPrimitive):
    """Computes the expanding trend for events over a given window.

    Description:
        Given a list of datetimes, return the expanding trend starting
        at the row `gap` rows away from the current row and looking backward over the specified
        time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. min_periods must always be less than or equal to the length of the series. (TODO: Check? Gap?)
            When window_length is an offset alias string, this limitation does not exist,
            but care should be taken to not choose a min_periods that will always be larger than the number of
            observations in the series.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and h.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.


    Examples:
        >>> import pandas as pd
        >>> expanding_trend = ExpandingTrend()
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> ans = expanding_trend(times, [5, 4, 3, 2, 1]).tolist()
        >>> [round(x, 2) for x in ans if not pd.isna(x) else x]
        [nan, nan, nan, -0.99, -0.99]

        We can also control the gap before the expanding calculation.

        >>> import pandas as pd
        >>> expanding_trend = ExpandingTrend(gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> expanding_trend(times, [5, 4, 3, 2, 1]).tolist()
        [5.0, 4.0, 3.0, 2.0, 1.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> expanding_trend = ExpandingTrend(min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> expanding_trend(times, [5, 4, 3, 2, 1]).tolist()
        [nan, nan, 4.0, 3.0, 2.0, 1.0]
    """

    name = "expanding_trend"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, gap=1, min_periods=1):
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def expanding_trend(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime)
            if isinstance(self.gap, int):
                x = x.shift(self.gap)
            else:
                raise NotImplementedError(
                    "We currently do not support string offsets for the gap parameter in "
                    "Expanding primitives",
                )
            return (
                x.expanding(min_periods=self.min_periods)
                .aggregate(calculate_trend)
                .values
            )

        return expanding_trend
