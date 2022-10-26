from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library


class TimeSinceLast(AggregationPrimitive):
    """Calculates the time elapsed since the last datetime (default in seconds).

    Description:
        Given a list of datetimes, calculate the
        time elapsed since the last datetime (default in
        seconds). Uses the instance's cutoff time.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> time_since_last = TimeSinceLast()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last(times, time=cutoff_time)
        150.0

        >>> from datetime import datetime
        >>> time_since_last = TimeSinceLast(unit = "minutes")
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last(times, time=cutoff_time)
        2.5

    """

    name = "time_since_last"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = True
    description_template = "the time since the last {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self, agg_type=Library.PANDAS):
        def time_since_last(values, time=None):
            time_since = time - values.iloc[-1]
            return convert_time_units(time_since.total_seconds(), self.unit)

        return time_since_last
