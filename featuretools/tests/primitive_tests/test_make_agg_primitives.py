import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

import featuretools as ft
from featuretools.primitives.base.aggregation_primitive_base import (
    make_agg_primitive
)


# Check the custom agg primitives description
def test_description_make_agg_primitive():
    def maximum(column):
        return max(column)

    Maximum = make_agg_primitive(function=maximum,
                                 input_types=[ColumnSchema(semantic_tags={'numeric'})],
                                 return_type=ColumnSchema(semantic_tags={'numeric'}))

    def maximum(column):
        '''Get the max value of a column.'''
        return max(column)

    Maximum2 = make_agg_primitive(function=maximum,
                                  input_types=[ColumnSchema(semantic_tags={'numeric'})],
                                  return_type=ColumnSchema(semantic_tags={'numeric'}),
                                  description='Get max value of a column.')

    Maximum3 = make_agg_primitive(function=maximum,
                                  input_types=[ColumnSchema(semantic_tags={'numeric'})],
                                  return_type=ColumnSchema(semantic_tags={'numeric'}),
                                  default_value=np.nan)

    assert Maximum.__doc__ != Maximum2.__doc__
    assert Maximum2.__doc__ != Maximum3.__doc__


# Check the successful default value for custom aggregation primitives


def test_default_value_make_agg_primitive(pd_mock_customer):

    def mean_sunday(numeric, datetime):
        '''
        Finds the mean of non-null values of a feature that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'numeric': numeric, 'time': days})
        return df[df['time'] == 6]['numeric'].mean()

    MeanSunday = make_agg_primitive(
        function=mean_sunday,
        input_types=[ColumnSchema(semantic_tags={'numeric'}), ColumnSchema(logical_type=Datetime)],
        return_type=ColumnSchema(semantic_tags={'numeric'})
    )

    feature_matrix, features = ft.dfs(entityset=pd_mock_customer,
                                      target_dataframe_name="sessions",
                                      agg_primitives=[MeanSunday],
                                      trans_primitives=[],
                                      max_depth=1)

    MeanSundayDefault = make_agg_primitive(
        function=mean_sunday,
        input_types=[ColumnSchema(semantic_tags={'numeric'}), ColumnSchema(logical_type=Datetime)],
        return_type=ColumnSchema(semantic_tags={'numeric'}),
        default_value=0
    )

    feature_matrix2, features = ft.dfs(entityset=pd_mock_customer,
                                       target_dataframe_name="sessions",
                                       agg_primitives=[MeanSundayDefault],
                                       trans_primitives=[],
                                       max_depth=1)

    assert (feature_matrix['MEAN_SUNDAY(transactions.amount, transaction_time)'].values != feature_matrix2['MEAN_SUNDAY(transactions.amount, transaction_time)'].values).any()
