from scipy.stats import kurtosis
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer

from featuretools.primitives.base import AggregationPrimitive


class Kurtosis(AggregationPrimitive):
    """Calculates the kurtosis for a list of numbers

    Args:
        fisher (bool): Optional. If True, Fisher's definition is used
            (normal ==> 0.0). If False, Pearson's definition is used
            (normal ==> 3.0). Default is True.
        bias (bool): Optional. If False, then the calculations are
            corrected for statistical bias. Default is True.
        nan_policy (str): Optional. Defines how to handle when
            input contains Nan. Possible values include
            `['propagate', 'raise', 'omit']`. 'propagate'
            returns Nan, 'raise' throws an error, 'omit'
            performs the calculations ignoring Nan values.
            Default is 'propagate'.

    Examples:
        >>> kurtosis = Kurtosis()
        >>> kurtosis([1, 2, 3, 4, 5])
        -1.3

        You can use Pearson's definition by setting the 'fisher' argument to False

        >>> kurtosis_fisher = Kurtosis(fisher=False)
        >>> kurtosis_fisher([1, 2, 3, 4, 5])
        1.7

        You can correct for statistical bias by setting the 'bias' argument to False

        >>> kurtosis_bias = Kurtosis(bias=False)
        >>> kurtosis_bias([1, 2, 3, 4, 5])
        -1.2000000000000004

        You can specifiy how to handle NaN values in the input with the 'nan_policy'
        argument

        >>> kurtosis_nan_policy = Kurtosis(nan_policy='omit')
        >>> kurtosis_nan_policy([1, 2, None, 3, 4, 5])
        -1.3
    """

    name = "kurtosis"
    input_types = [
        [ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})],
        [ColumnSchema(logical_type=Double, semantic_tags={"numeric"})],
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, fisher=True, bias=True, nan_policy="propagate"):
        if nan_policy not in ["propagate", "raise", "omit"]:
            raise ValueError("Invalid nan_policy")
        self.fisher = fisher
        self.bias = bias
        self.nan_policy = nan_policy

    def get_function(self):
        def kurtosis_func(x):
            return kurtosis(
                x,
                axis=0,
                fisher=self.fisher,
                bias=self.bias,
                nan_policy=self.nan_policy,
            )

        return kurtosis_func
