import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Timedelta

from featuretools.primitives.base.transform_primitive_base import (
    make_trans_primitive
)


# Check the custom trans primitives description
def test_description_make_trans_primitives():
    def pd_time_since(array, moment):
        return (moment - pd.DatetimeIndex(array)).values

    TimeSince = make_trans_primitive(
        function=pd_time_since,
        input_types=[
            [ColumnSchema(logical_type=Datetime, semantic_tabs={'time_index'})],
            [ColumnSchema(logical_type=Datetime)]
        ],
        return_type=ColumnSchema(logical_type=Timedelta),
        uses_calc_time=True,
        name="time_since"
    )

    def pd_time_since(array, moment):
        """Calculates time since the cutoff time."""
        return (moment - pd.DatetimeIndex(array)).values

    TimeSince2 = make_trans_primitive(
        function=pd_time_since,
        input_types=[
            [ColumnSchema(logical_type=Datetime, semantic_tabs={'time_index'})],
            [ColumnSchema(logical_type=Datetime)]
        ],
        return_type=ColumnSchema(logical_type=Timedelta),
        uses_calc_time=True,
        name="time_since"
    )

    TimeSince3 = make_trans_primitive(
        function=pd_time_since,
        input_types=[
            [ColumnSchema(logical_type=Datetime, semantic_tabs={'time_index'})],
            [ColumnSchema(logical_type=Datetime)]
        ],
        return_type=ColumnSchema(logical_type=Timedelta),
        description="Calculates time since the cutoff time.",
        name="time_since"
    )

    assert TimeSince.__doc__ != TimeSince2.__doc__
    assert TimeSince2.__doc__ == TimeSince3.__doc__
