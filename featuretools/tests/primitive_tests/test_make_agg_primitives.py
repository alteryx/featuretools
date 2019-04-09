import numpy as np
import pandas as pd

import featuretools as ft
from featuretools.primitives.base.aggregation_primitive_base import (
    make_agg_primitive
)
from featuretools.variable_types import Datetime, Numeric


# Check the custom agg primitives description
def test_description_make_agg_primitive():
    def maximum(column):
        return max(column)

    Maximum = make_agg_primitive(function=maximum,
                                 input_types=[Numeric],
                                 return_type=Numeric)

    def maximum(column):
        '''Get the max value of a column.'''
        return max(column)

    Maximum2 = make_agg_primitive(function=maximum,
                                  input_types=[Numeric],
                                  return_type=Numeric,
                                  description='Get max value of a column.')

    Maximum3 = make_agg_primitive(function=maximum,
                                  input_types=[Numeric],
                                  return_type=Numeric,
                                  default_value=np.nan)

    assert Maximum.__doc__ != Maximum2.__doc__
    assert Maximum2.__doc__ != Maximum3.__doc__


# Check the successful default value for custom aggregation primitives
def test_default_value_make_agg_primitive():
    es = ft.demo.load_mock_customer(return_entityset=True)

    def mean_sunday(numeric, datetime):
        '''
        Finds the mean of non-null values of a feature that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'numeric': numeric, 'time': days})
        return df[df['time'] == 6]['numeric'].mean()

    MeanSunday = make_agg_primitive(function=mean_sunday,
                                    input_types=[Numeric, Datetime],
                                    return_type=Numeric)

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=[MeanSunday],
                                      trans_primitives=[],
                                      max_depth=1)

    MeanSundayDefault = make_agg_primitive(function=mean_sunday,
                                           input_types=[Numeric, Datetime],
                                           return_type=Numeric,
                                           default_value=0)

    feature_matrix2, features = ft.dfs(entityset=es,
                                       target_entity="sessions",
                                       agg_primitives=[MeanSundayDefault],
                                       trans_primitives=[],
                                       max_depth=1)

    assert (feature_matrix['MEAN_SUNDAY(transactions.amount, transaction_time)'].values != feature_matrix2['MEAN_SUNDAY(transactions.amount, transaction_time)'].values).any()
