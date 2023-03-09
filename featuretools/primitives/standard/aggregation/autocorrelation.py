from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import AggregationPrimitive


class AutoCorrelation(AggregationPrimitive):
    """Determines the Pearson correlation between a series and a shifted
    version of the series.

    Args:
        lag (int): The number of periods to shift the input before performing
            correlation. Default is 1.

    Examples:
        >>> autocorrelation = AutoCorrelation()
        >>> round(autocorrelation([1, 2, 3, 1, 3, 2]), 3)
        -0.598

        The number of periods to shift the input before performing correlation
        can be specified.

        >>> autocorrelation_lag = AutoCorrelation(lag=3)
        >>> autocorrelation_lag([1, 2, 3, 1, 2, 3])
        1.0
    """

    name = "auto_correlation"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, lag=1):
        self.lag = lag

    def get_function(self):
        def auto_correlation(x):
            return x.autocorr(lag=self.lag)

        return auto_correlation
