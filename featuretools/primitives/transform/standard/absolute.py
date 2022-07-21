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


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number.

    Examples:
        >>> absolute = Absolute()
        >>> absolute([3.0, -5.0, -2.4]).tolist()
        [3.0, 5.0, 2.4]
    """

    name = "absolute"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the absolute value of {}"

    def get_function(self):
        return np.absolute
