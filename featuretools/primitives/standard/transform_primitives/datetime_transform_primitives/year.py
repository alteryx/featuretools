from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Ordinal

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Year(TransformPrimitive):
    """Determines the year value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2048, 6, 17, 11, 10, 50),
        ...          datetime(1950, 11, 30, 19, 45, 15)]
        >>> year = Year()
        >>> year(dates).tolist()
        [2019, 2048, 1950]
    """

    name = "year"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(
        logical_type=Ordinal(order=list(range(1, 3000))),
        semantic_tags={"category"},
    )
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the year of {}"

    def get_function(self):
        def year(vals):
            return vals.dt.year

        return year
