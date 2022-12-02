from scipy import stats
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive


class BoxCox(TransformPrimitive):
    """Applies a transformation to help your data resemble a normal distribution,
        commonly known as the Box Cox transformation. The input data must be positive.
        It cannot contain zeroes or negative numbers.
        Commonly used with time series data, to reduce the variance.

    Args:
        lmbda (int, float, optional): The exponent to use for the transformation.
            Default is None.

    Examples:
        >>> box_cox = BoxCox()
        >>> transformed = box_cox([1, 10, 5, 4, 2])
        >>> transformed = [round(x, 2) for x in transformed.tolist()]
        >>> transformed
        [0.0, 2.69, 1.79, 1.52, 0.73]

        You can specify the lambda to use for the tranformation

        >>> box_cox = BoxCox(lmbda=-1)
        >>> transformed = box_cox([1, 10, 5, 4, 2])
        >>> transformed = transformed.tolist()
        >>> transformed
        [0.0, 0.9, 0.8, 0.75, 0.5]

    """

    name = "box_cox"
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
                return stats.boxcox(numeric, lmbda=self.lmbda)
            return stats.boxcox(numeric)[0]

        return box_cox
