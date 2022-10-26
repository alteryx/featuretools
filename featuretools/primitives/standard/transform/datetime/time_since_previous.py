from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils import convert_time_units


class TimeSincePrevious(TransformPrimitive):
    """Computes the time since the previous entry in a list.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to Seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Description:
        Given a list of datetimes, compute the time in seconds elapsed since
        the previous item in the list. The result for the first item in the
        list will always be `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_previous = TimeSincePrevious()
        >>> dates = [datetime(2019, 3, 1, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0),
        ...          datetime(2019, 3, 1, 0, 3, 0),
        ...          datetime(2019, 3, 1, 0, 2, 30),
        ...          datetime(2019, 3, 1, 0, 10, 0)]
        >>> time_since_previous(dates).tolist()
        [nan, 120.0, 60.0, -30.0, 450.0]
    """

    name = "time_since_previous"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the time since the previous instance of {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self):
        def pd_diff(values):
            return convert_time_units(
                values.diff().apply(lambda x: x.total_seconds()),
                self.unit,
            )

        return pd_diff
