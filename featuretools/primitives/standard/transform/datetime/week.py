from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Ordinal

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Week(TransformPrimitive):
    """Determines the week of the year from a datetime.

    Description:
        Returns the week of the year from a datetime value. The first week
        of the year starts on January 1, and week numbers increment each
        Monday.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 1, 3),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> week = Week()
        >>> week(dates).tolist()
        [1, 25, 48]
    """

    name = "week"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(1, 54))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the week of the year of {}"

    def get_function(self):
        def week(vals):
            if hasattr(vals.dt, "isocalendar"):
                return vals.dt.isocalendar().week
            else:
                return vals.dt.week

        return week
