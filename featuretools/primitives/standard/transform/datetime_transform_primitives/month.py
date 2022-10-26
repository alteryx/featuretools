from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Ordinal

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Month(TransformPrimitive):
    """Determines the month value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> month = Month()
        >>> month(dates).tolist()
        [3, 6, 11]
    """

    name = "month"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(1, 13))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the month of {}"

    def get_function(self):
        def month(vals):
            return vals.dt.month

        return month
