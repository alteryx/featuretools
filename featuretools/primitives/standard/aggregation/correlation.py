import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import AggregationPrimitive


class Correlation(AggregationPrimitive):
    """Computes the correlation between two columns of values.

    Args:
        method (str): Method to compute correlation. Options
            include the following {‘pearson’, ‘kendall’, ‘spearman’}.
            Defaults to `pearson`.
            - ``pearson``, Standard correlation coefficient
            - ``kendall``, Kendall Tau correlation coefficient
            - ``spearman``, Spearman rank correlation coefficient

    Examples:
        >>> correlation = Correlation()
        >>> array_1 = [1, 4, 6, 7]
        >>> array_2 = [1, 5, 9, 7]
        >>> correlation(array_1, array_2)
        0.9221388919541468

        We can also use different methods of computation.

        >>> correlation_pearson = Correlation(method='pearson')
        >>> correlation_pearson(array_1, array_2)
        0.9221388919541468
        >>> correlation_spearman = Correlation(method='spearman')
        >>> correlation_spearman(array_1, array_2)
        0.7999999999999999
        >>> correlation_kendall = Correlation(method='kendall')
        >>> correlation_kendall(array_1, array_2)
        0.6666666666666669
    """

    name = "correlation"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = np.nan

    def __init__(self, method="pearson"):
        acceptable_methods = ["pearson", "spearman", "kendall"]
        if method not in acceptable_methods:
            raise ValueError(
                "Invalid method, valid methods are %s"
                % (", ".join(acceptable_methods)),
            )
        self.method = method

    def get_function(self):
        def correlation(col_1, col_2):
            return col_1.corr(col_2, method=self.method)

        return correlation
