from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsWorkingHours(TransformPrimitive):
    """Determines if a datetime falls during working hours on a 24-hour clock. Can configure start_hour and end_hour.

    Args:
        start_hour (int): Start hour of workday. Must adhere to 24-hour clock. Default is 8 (8am).
        end_hour (int): End hour of workday. Must adhere to 24-hour clock. Default is 18 (6pm).

    Examples:
        >>> import numpy as np
        >>> from datetime import datetime
        >>> dates = [datetime(2022, 6, 21, 16, 3, 3),
        ...          datetime(2019, 1, 3, 4, 4, 4),
        ...          datetime(2022, 1, 1, 12, 1, 2),
        ...          np.nan]
        >>> is_working_hour = IsWorkingHours()
        >>> is_working_hour(dates).tolist()
        [True, False, True, False]
        >>> is_working_hour = IsWorkingHours(15, 17)
        >>> is_working_hour(dates).tolist()
        [True, False, False, False]
    """

    name = "is_working_hours"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} falls during working hours"

    def __init__(self, start_hour=8, end_hour=18):
        self.start_hour = start_hour
        self.end_hour = end_hour

    def get_function(self):
        def is_working_hours(vals):
            return (vals.dt.hour >= self.start_hour) & (vals.dt.hour <= self.end_hour)

        return is_working_hours
