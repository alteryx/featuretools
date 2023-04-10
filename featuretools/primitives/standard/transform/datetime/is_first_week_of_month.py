import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive


class IsFirstWeekOfMonth(TransformPrimitive):
    """Determines if a date falls in the first week of the month.

    Description:
        Converts a datetime to a boolean indicating if the date
        falls in the first week of the month. The first week of
        the month starts on day 1, and the week number is incremented
        each Sunday.

    Examples:
        >>> from datetime import datetime
        >>> is_first_week_of_month = IsFirstWeekOfMonth()
        >>> times = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3),
        ...          datetime(2019, 3, 31),
        ...          datetime(2019, 3, 30)]
        >>> is_first_week_of_month(times).tolist()
        [True, False, False, False]
    """

    name = "is_first_week_of_month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)

    def get_function(self):
        def is_first_week_of_month(x):
            df = pd.DataFrame({"date": x})
            df["first_day"] = df.date - pd.to_timedelta(df["date"].dt.day - 1, unit="d")
            df["dom"] = df.date.dt.day
            df["first_day_weekday"] = df.first_day.dt.weekday
            df["adjusted_dom"] = df.dom + df.first_day_weekday + 1
            df.loc[df["first_day_weekday"].astype(float) == 6.0, "adjusted_dom"] = df[
                "dom"
            ]
            df["is_first_week"] = np.ceil(df.adjusted_dom / 7.0) == 1.0
            if df["date"].isnull().values.any():
                df["is_first_week"] = df["is_first_week"].astype("object")
                df.loc[df["date"].isnull(), "is_first_week"] = np.nan
            return df.is_first_week.values

        return is_first_week_of_month
