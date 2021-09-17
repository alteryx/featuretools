import pandas as pd
import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Datetime, Timedelta

from featuretools.feature_base import Feature
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
            [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})],
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
            [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})],
            [ColumnSchema(logical_type=Datetime)]
        ],
        return_type=ColumnSchema(logical_type=Timedelta),
        uses_calc_time=True,
        name="time_since"
    )

    TimeSince3 = make_trans_primitive(
        function=pd_time_since,
        input_types=[
            [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})],
            [ColumnSchema(logical_type=Datetime)]
        ],
        return_type=ColumnSchema(logical_type=Timedelta),
        description="Calculates time since the cutoff time.",
        name="time_since"
    )

    assert TimeSince.__doc__ != TimeSince2.__doc__
    assert TimeSince2.__doc__ == TimeSince3.__doc__


def test_make_transform_restricts_time_keyword():
    make_trans_primitive(
        lambda x, time=False: x,
        [ColumnSchema(logical_type=Datetime)],
        ColumnSchema(semantic_tags={'numeric'}),
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda x, time=False: x,
            [ColumnSchema(logical_type=Datetime)],
            ColumnSchema(semantic_tags={'numeric'}),
            name="BadPrimitive",
            description="This primitive should error")


def test_make_transform_restricts_time_arg():
    make_trans_primitive(
        lambda time: time,
        [ColumnSchema(logical_type=Datetime)],
        ColumnSchema(semantic_tags={'numeric'}),
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda time: time,
            [ColumnSchema(logical_type=Datetime)],
            ColumnSchema(semantic_tags={'numeric'}),
            name="BadPrimitive",
            description="This primitive should erorr")


def test_make_transform_sets_kwargs_correctly(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [ColumnSchema()],
        ColumnSchema(logical_type=Boolean),
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin_1_list = ["toothpaste", "coke_zero"]
    isin_1_base_f = Feature(es['log'].ww['product_id'])
    isin_1 = Feature(isin_1_base_f, primitive=IsIn(list_of_outputs=isin_1_list))
    isin_2_list = ["coke_zero"]
    isin_2_base_f = Feature(es['log'].ww['session_id'])
    isin_2 = Feature(isin_2_base_f, primitive=IsIn(list_of_outputs=isin_2_list))
    assert isin_1_base_f == isin_1.base_features[0]
    assert isin_1_list == isin_1.primitive.kwargs['list_of_outputs']
    assert isin_2_base_f == isin_2.base_features[0]
    assert isin_2_list == isin_2.primitive.kwargs['list_of_outputs']
