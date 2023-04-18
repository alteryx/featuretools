from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Ordinal

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class DaysInMonth(TransformPrimitive):
    """Determines the number of days in the month of given datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 12, 1),
        ...          datetime(2019, 1, 3),
        ...          datetime(2020, 2, 1)]
        >>> days_in_month = DaysInMonth()
        >>> days_in_month(dates).tolist()
        [31, 31, 29]
    """

    name = "days_in_month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(1, 32))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the days in the month of {}"

    def get_function(self):
        def days_in_month(vals):
            return vals.dt.daysinmonth

        return days_in_month
