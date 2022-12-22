import numpy as np
import pandas as pd
from woodwork import init_series
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.standard.transform.time_series.utils import (
    apply_rolling_agg_to_series,
)


class RollingOutlierCount(TransformPrimitive):
    """Determines how many values are outliers over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling count of outliers within the numeric values,
        starting at the row `gap` rows away from the current row and looking backward
        over the specified window (by `window_length` and `gap`). Values are deemed
        outliers using the IQR method, computed over the whole series.
        Input datetimes should be monotonic.

    Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling
            frequency, for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of Pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of Pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1, which excludes the target instance from the window.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months are different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_outlier_count = RollingOutlierCount(window_length=4)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=6)
        >>> rolling_outlier_count(times, [0, 0, 0, 0, 10, 0]).tolist()
        [nan, 0.0, 0.0, 0.0, 0.0, 1.0]

        We can also control the gap before the rolling calculation.
        >>> import pandas as pd
        >>> rolling_outlier_count = RollingOutlierCount(window_length=4, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=6)
        >>> rolling_outlier_count(times, [0, 0, 0, 0, 10, 0]).tolist()
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        We can also control the minimum number of periods required for the rolling calculation.
        >>> import pandas as pd
        >>> rolling_outlier_count = RollingOutlierCount(window_length=4, min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=6)
        >>> rolling_outlier_count(times,  [0, 0, 0, 0, 10, 0]).tolist()
        [nan, nan, nan, 0.0, 0.0, 1.0]

        We can also set the window_length and gap using offset alias strings.
        >>> import pandas as pd
        >>> rolling_outlier_count = RollingOutlierCount(window_length='4min', gap='1min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=6)
        >>> rolling_outlier_count(times, [0, 0, 0, 0, 10, 0]).tolist()
        [nan, 0.0, 0.0, 0.0, 0.0, 1.0]
    """

    name = "rolling_outlier_count"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_full_dataframe = True

    def __init__(self, window_length=3, gap=1, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_outliers_count(self, numeric_series):
        # We know the column is numeric, so use the Double logical type in case Woodwork's
        # type inference could not infer a numeric type
        if not len(numeric_series.dropna()):
            return np.nan
        if numeric_series.ww.schema is None:
            numeric_series = init_series(numeric_series, logical_type="Double")
        box_plot_info = numeric_series.ww.box_plot_dict()
        return len(box_plot_info["high_values"]) + len(box_plot_info["low_values"])

    def get_function(self):
        def rolling_outlier_count(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                series=x,
                agg_func=self.get_outliers_count,
                window_length=self.window_length,
                gap=self.gap,
                min_periods=self.min_periods,
                ignore_window_nans=False,
            )

        return rolling_outlier_count
