from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive


class ExponentialWeightedVariance(TransformPrimitive):
    """Computes the exponentially weighted moving variance for a series of numbers

    Description:
        Returns the exponentially weighted moving variance for a series of
        numbers. Exactly one of center of mass (com), span, half-life, and
        alpha must be provided. Missing values can be ignored when calculating
        weights by setting 'ignore_na' to True.

    Args:
        com (float): Specify decay in terms of center of mass for com >= 0.
            Default is None.

        span (float): Specify decay in terms of span for span >= 1.
            Default is None.

        halflife (float): Specify decay in terms of half-life for halflife > 0.
            Default is None.

        alpha (float): Specify smoothing factor alpha directly. Alpha should be
            greater than 0 and less than or equal to 1. Default is None.

        ignore_na (bool): Ignore missing values when calculating weights.
            Default is False.

    Examples:
        >>> exponential_weighted_variance = ExponentialWeightedVariance(com=0.5)
        >>> exponential_weighted_variance([1, 2, 3, 4]).tolist()
        [nan, 0.49999999999999983, 0.8461538461538459, 1.1230769230769233]

        Missing values can be ignored

        >>> ewmv_ignorena = ExponentialWeightedVariance(com=0.5, ignore_na=True)
        >>> ewmv_ignorena([1, 2, 3, None, 4]).tolist()
        [nan, 0.49999999999999983, 0.8461538461538459, 0.8461538461538459, 1.1230769230769233]
    """

    name = "exponential_weighted_variance"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_full_dataframe = True

    def __init__(self, com=None, span=None, halflife=None, alpha=None, ignore_na=False):
        if all(x is None for x in [com, span, halflife, alpha]):
            com = 0.5
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.ignore_na = ignore_na

    def get_function(self):
        def exponential_weighted_average(x):
            return x.ewm(
                com=self.com,
                span=self.span,
                halflife=self.halflife,
                alpha=self.alpha,
                ignore_na=self.ignore_na,
            ).var()

        return exponential_weighted_average
