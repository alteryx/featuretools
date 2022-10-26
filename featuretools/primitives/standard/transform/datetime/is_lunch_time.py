from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsLunchTime(TransformPrimitive):
    """Determines if a datetime falls during configurable lunch hour, on a 24-hour clock.

    Args:
        lunch_hour (int): Hour when lunch is taken. Must adhere to 24-hour clock. Defaults to 12.

    Examples:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> dates = [datetime(2022, 6, 21, 12, 3, 3),
        ...          datetime(2019, 1, 3, 4, 4, 4),
        ...          datetime(2022, 1, 1, 11, 1, 2),
        ...          np.nan]
        >>> is_lunch_time = IsLunchTime()
        >>> is_lunch_time(dates).tolist()
        [True, False, False, False]
        >>> is_lunch_time = IsLunchTime(11)
        >>> is_lunch_time(dates).tolist()
        [False, False, True, False]
    """

    name = "is_lunch_time"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} falls during lunch time"

    def __init__(self, lunch_hour=12):
        self.lunch_hour = lunch_hour

    def get_function(self):
        def is_lunch_time(vals):
            return vals.dt.hour == self.lunch_hour

        return is_lunch_time
