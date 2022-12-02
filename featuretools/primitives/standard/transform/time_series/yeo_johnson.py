import pandas as pd
from scipy import stats
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive


class YeoJohnson(TransformPrimitive):
    """Applies a transformation to help your data resemble a normal distribution,
        commonly known as the Yeo-Johnson power transformation. Works with negative,
        positive, and zero values. Often used with time series data, to reduce the variance.

    Args:
        lmbda (int, float, optional): The exponent to use for the transformation.
            Default is None.

    Examples:
        >>> yeo_johnson = YeoJohnson()
        >>> transformed = yeo_johnson([1, -10, 5, -4, 2])
        >>> transformed = [round(x, 2) for x in transformed.tolist()]
        >>> transformed
        [1.12, -6.26, 7.1, -2.99, 2.43]

        You can specify the lambda to use for the transformation

        >>> yeo_johnson = YeoJohnson(lmbda=-1)
        >>> transformed = yeo_johnson([1, -10, 5, -4, 2])
        >>> transformed = [round(x, 2) for x in transformed.tolist()]
        >>> transformed
        [0.5, -443.33, 0.83, -41.33, 0.67]

    """

    name = "yeo_johnson"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_full_dataframe = True

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def get_function(self):
        def box_cox(numeric):
            if self.lmbda is not None:
                return stats.yeojohnson(numeric, lmbda=self.lmbda)
            return stats.yeojohnson(numeric)[0]

        return box_cox
