from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsYearEnd(TransformPrimitive):
    """Determines if a date falls on the end of a year.

    Examples:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 12, 31),
        ...          datetime(2019, 1, 1),
        ...          datetime(2019, 11, 30),
        ...          np.nan]
        >>> is_year_end = IsYearEnd()
        >>> is_year_end(dates).tolist()
        [True, False, False, False]
    """

    name = "is_year_end"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} occurred on the end of a year"

    def get_function(self):
        def is_year_end(vals):
            return vals.dt.is_year_end

        return is_year_end
