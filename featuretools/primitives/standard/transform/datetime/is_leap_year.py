from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsLeapYear(TransformPrimitive):
    """Determines the is_leap_year attribute of a datetime column.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2020, 3, 3, 11, 10, 50),
        ...          datetime(2021, 3, 31, 19, 45, 15)]
        >>> ily = IsLeapYear()
        >>> ily(dates).tolist()
        [False, True, False]
    """

    name = "is_leap_year"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether the year of {} is a leap year"

    def get_function(self):
        def is_leap_year(vals):
            return vals.dt.is_leap_year

        return is_leap_year
