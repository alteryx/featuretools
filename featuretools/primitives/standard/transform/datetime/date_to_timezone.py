import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime

from featuretools.primitives.base import TransformPrimitive


class DateToTimeZone(TransformPrimitive):
    """Determines the timezone of a datetime.

    Description:
        Given a list of datetimes, extract the timezone from each
        one. Looks for the `tzinfo` attribute on `datetime.datetime`
        objects. If the datetime has no timezone or the date is
        missing, return `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> from pytz import timezone
        >>> date_to_time_zone = DateToTimeZone()
        >>> dates = [datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
        ...          datetime(2010, 1, 1, tzinfo=timezone("America/New_York")),
        ...          datetime(2010, 1, 1, tzinfo=timezone("America/Chicago")),
        ...          datetime(2010, 1, 1)]
        >>> date_to_time_zone(dates).tolist()
        ['America/Los_Angeles', 'America/New_York', 'America/Chicago', nan]
    """

    name = "date_to_time_zone"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def date_to_time_zone(x):
            return x.apply(lambda x: x.tzinfo.zone if x.tzinfo else np.nan)

        return date_to_time_zone
