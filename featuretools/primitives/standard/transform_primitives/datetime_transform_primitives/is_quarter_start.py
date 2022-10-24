from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsQuarterStart(TransformPrimitive):
    """Determines the is_quarter_start attribute of a datetime column.

    Examples:
        >>> from datetime import datetime
        >>> iqs = IsQuarterStart()
        >>> dates = [datetime(2020, 3, 31),
        ...          datetime(2020, 1, 1)]
        >>> iqs(dates).tolist()
        [False, True]
    """

    name = "is_quarter_start"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is a quarter start"

    def get_function(self):
        def is_quarter_start(vals):
            return vals.dt.is_quarter_start

        return is_quarter_start
