from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class IsQuarterEnd(TransformPrimitive):
    """Determines the is_quarter_end attribute of a datetime column.

    Examples:
        >>> from datetime import datetime
        >>> iqe = IsQuarterEnd()
        >>> dates = [datetime(2020, 3, 31),
        ...          datetime(2020, 1, 1)]
        >>> iqe(dates).tolist()
        [True, False]
    """

    name = "is_quarter_end"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is a quarter end"

    def get_function(self):
        def is_quarter_end(vals):
            return vals.dt.is_quarter_end

        return is_quarter_end
