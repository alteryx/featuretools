from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Ordinal

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Second(TransformPrimitive):
    """Determines the seconds value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> second = Second()
        >>> second(dates).tolist()
        [0, 50, 15]
    """

    name = "second"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(60))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the seconds value of {}"

    def get_function(self):
        def second(vals):
            return vals.dt.second

        return second
