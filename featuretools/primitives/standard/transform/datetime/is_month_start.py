from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsMonthStart(TransformPrimitive):
    """Determines the is_month_start attribute of a datetime column.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2020, 2, 13),
        ...          datetime(2020, 2, 29)]
        >>> ims = IsMonthStart()
        >>> ims(dates).tolist()
        [True, False, False]
    """

    name = "is_month_start"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is at the start of a month"

    def get_function(self):
        def is_month_start(vals):
            return vals.dt.is_month_start

        return is_month_start
