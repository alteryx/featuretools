from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive


class PercentChange(TransformPrimitive):
    """Determines the percent difference between values in a list.

    Description:
        Given a list of numbers, return the percent difference
        between each subsequent number. Percentages are shown in
        decimal form (not multiplied by 100). Uses pandas' pct_change
        function.

    Args:
        periods (int): Periods to shift for calculating percent change.
            Default is 1.

        fill_method (str): Method for filling gaps in reindexed
            Series. Valid options are `backfill`, `bfill`, `pad`, `ffill`.
            `pad / ffill`: fill gap with last valid observation.
            `backfill / bfill`: fill gap with next valid observation.
            Default is `pad`.

        limit (int): The max number of consecutive NaN values in a gap that
            can be filled. Default is None.

        freq (DateOffset, timedelta, or offset alias string):
            If `freq` is specified, instead of calcualting change between subsequent
            points, PercentChange will calculate change between points with a
            certain interval between their date indices. `freq` defines the
            desired interval. When freq is used, the resulting index will also be
            filled to include any missing dates from the specified interval.

            If the index is not date/datetime and freq is used, it will raise a
            NotImplementedError.

            If freq is None, no changes will be applied. Default is None.

    Examples:
        >>> percent_change = PercentChange()
        >>> percent_change([2, 5, 15, 3, 3, 9, 4.5]).to_list()
        [nan, 1.5, 2.0, -0.8, 0.0, 2.0, -0.5]

        We can control the number of periods to return the percent
            difference between points further from one another.

        >>> percent_change_2 = PercentChange(periods=2)
        >>> percent_change_2([2, 5, 15, 3, 3, 9, 4.5]).to_list()
        [nan, nan, 6.5, -0.4, -0.8, 2.0, 0.5]

        We can control the method used to handle gaps in data.

        >>> percent_change = PercentChange()
        >>> percent_change([2, 4, 8, None, 16, None, 32, None]).to_list()
        [nan, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        >>> percent_change_backfill = PercentChange(fill_method='backfill')
        >>> percent_change_backfill([2, 4, 8, None, 16, None, 32, None]).to_list()
        [nan, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, nan]

        We can also control the maximum number of NaN values to fill in a gap.

        >>> percent_change = PercentChange()
        >>> percent_change([2, None, None, None, 4]).to_list()
        [nan, 0.0, 0.0, 0.0, 1.0]
        >>> percent_change_limited = PercentChange(limit=2)
        >>> percent_change_limited([2, None, None, None, 4]).to_list()
        [nan, 0.0, 0.0, nan, nan]

        Finally, we can specify a date frequency on which to calculate percent
            change.

        >>> import pandas as pd
        >>> dates = pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-05'])
        >>> x_indexed = pd.Series([1, 2, 3, 4], index=dates)
        >>> percent_change = PercentChange()
        >>> percent_change(x_indexed).to_list()
        [nan, 1.0, 0.5, 0.33333333333333326]
        >>> date_offset = pd.tseries.offsets.DateOffset(days=1)
        >>> percent_change_freq = PercentChange(freq=date_offset)
        >>> percent_change_freq(x_indexed).to_list()
        [nan, 1.0, 0.5, nan]
    """

    name = "percent_change"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, periods=1, fill_method="pad", limit=None, freq=None):
        if fill_method not in ["backfill", "bfill", "pad", "ffill"]:
            raise ValueError("Invalid fill_method")
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.freq = freq

    def get_function(self):
        def percent_change(data):
            return data.pct_change(
                self.periods,
                self.fill_method,
                self.limit,
                self.freq,
            )

        return percent_change
