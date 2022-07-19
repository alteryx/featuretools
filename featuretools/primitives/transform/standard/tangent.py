import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    URL,
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    NaturalLanguage,
    Timedelta,
)

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.utils.common_tld_utils import COMMON_TLDS
from featuretools.utils.gen_utils import Library

class Tangent(TransformPrimitive):
    """Computes the tangent of a number.

    Examples:
        >>> tan = Tangent()
        >>> tan([-np.pi, 0.0, np.pi/2.0]).tolist()
        [1.2246467991473532e-16, 0.0, 1.633123935319537e+16]
    """

    name = "tangent"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the tangent of {}"

    def get_function(self):
        return np.tan