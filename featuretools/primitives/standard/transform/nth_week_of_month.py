import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base import TransformPrimitive


class NthWeekOfMonth(TransformPrimitive):
    """Determines the nth week of the month from a given date.

    Description:
        Converts a datetime to an float representing the week
        of the month in which the date falls. The first day of
        the month starts week 1, and the week number is incremented
        each Sunday.

    Examples:
        >>> from datetime import datetime
        >>> nth_week_of_month = NthWeekOfMonth()
        >>> times = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3),
        ...          datetime(2019, 3, 31),
        ...          datetime(2019, 3, 30)]
        >>> nth_week_of_month(times).tolist()
        [1.0, 2.0, 6.0, 5.0]
    """

    name = "nth_week_of_month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def get_function(self):
        def nth_week_of_month(x):
            df = pd.DataFrame({"date": x})
            df["first_day"] = df.date - pd.to_timedelta(df["date"].dt.day - 1, unit="d")
            df["dom"] = df.date.dt.day
            df["first_day_weekday"] = df.first_day.dt.weekday
            df["adjusted_dom"] = df.dom + df.first_day_weekday + 1
            df.loc[df["first_day_weekday"].astype(float) == 6.0, "adjusted_dom"] = df[
                "dom"
            ]
            df["week_of_month"] = np.ceil(df.adjusted_dom / 7.0)
            return df.week_of_month.values

        return nth_week_of_month
