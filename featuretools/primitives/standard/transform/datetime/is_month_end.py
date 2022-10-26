from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsMonthEnd(TransformPrimitive):
    """Determines the is_month_end attribute of a datetime column.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2021, 2, 28),
        ...          datetime(2020, 2, 29)]
        >>> ime = IsMonthEnd()
        >>> ime(dates).tolist()
        [False, True, True]
    """

    name = "is_month_end"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is at the end of a month"

    def get_function(self):
        def is_month_end(vals):
            return vals.dt.is_month_end

        return is_month_end
