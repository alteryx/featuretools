import holidays
import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    AgeFractional,
    BooleanNullable,
    Categorical,
    Datetime,
    Ordinal,
)

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.primitives.utils import HolidayUtil
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library

class Quarter(TransformPrimitive):
    """Determines the quarter a datetime column falls into (1, 2, 3, 4)

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019,12,1),
        ...          datetime(2019,1,3),
        ...          datetime(2020,2,1)]
        >>> q = Quarter()
        >>> q(dates).tolist()
        [4, 1, 1]
    """

    name = "quarter"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(1, 5))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the quarter that describes {}"

    def get_function(self):
        def quarter(vals):
            return vals.dt.quarter

        return quarter