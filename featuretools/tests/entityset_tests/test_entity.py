from datetime import datetime

# import numpy as np
import pandas as pd
import pytest
import woodwork as ww

import featuretools as ft
# from featuretools.entityset import Entity, EntitySet
from featuretools.tests.testing_utils import (
    make_ecommerce_entityset,
    to_pandas
)
from featuretools.utils.gen_utils import Library


def test_reorders_index():
    es = ft.EntitySet('test')
    df = pd.DataFrame({'id': [1, 2, 3], 'other': [4, 5, 6]})
    df.columns = ['other', 'id']
    es.add_dataframe(dataframe_name='test', dataframe=df, index='id')
    assert es['test'].variables[0].id == 'id'
    assert es['test'].variables[0].id == es['test'].index
    assert [v.id for v in es['test'].variables] == list(es['test'].df.columns)


def test_eq(es):
    other_es = make_ecommerce_entityset()
    latlong = es['log'].df['latlong'].copy()

    assert es['log'].__eq__(es['log'], deep=True)
    assert es['log'].__eq__(other_es['log'], deep=True)
    assert all(to_pandas(es['log'].df['latlong']).eq(to_pandas(latlong)))

    # Test different index
    other_es['log'].index = None
    assert not es['log'].__eq__(other_es['log'])
    other_es['log'].index = 'id'
    assert es['log'].__eq__(other_es['log'])

    # Test different time index
    other_es['log'].time_index = None
    assert not es['log'].__eq__(other_es['log'])
    other_es['log'].time_index = 'datetime'
    assert es['log'].__eq__(other_es['log'])

    # Test different secondary time index
    other_es['customers'].secondary_time_index = {}
    assert not es['customers'].__eq__(other_es['customers'])
    other_es['customers'].secondary_time_index = {
        'cancel_date': ['cancel_reason', 'cancel_date']}
    assert es['customers'].__eq__(other_es['customers'])

    original_variables = es['sessions'].variables
    # Test different variable list length
    other_es['sessions'].variables = original_variables[:-1]
    assert not es['sessions'].__eq__(other_es['sessions'])
    # Test different variable list contents
    other_es['sessions'].variables = original_variables[:-1] + [original_variables[0]]
    assert not es['sessions'].__eq__(other_es['sessions'])

    # Test different interesting values
    assert es['log'].__eq__(other_es['log'], deep=True)
    other_es.add_interesting_values(entity_id='log')
    assert not es['log'].__eq__(other_es['log'], deep=True)

    # Check one with last time index, one without
    other_es['log'].last_time_index = other_es['log'].df['datetime']
    assert not other_es['log'].__eq__(es['log'], deep=True)
    assert not es['log'].__eq__(other_es['log'], deep=True)
    # Both set with different values
    es['log'].last_time_index = other_es['log'].last_time_index + pd.Timedelta('1h')
    assert not other_es['log'].__eq__(es['log'], deep=True)

    # Check different dataframes
    other_es['stores'].df = other_es['stores'].df.head(0)
    assert not other_es['stores'].__eq__(es['stores'], deep=True)


def test_update_dataframe(es):
    df = es['customers'].df.copy()
    if es.dataframe_type == Library.KOALAS.value:
        df['new'] = [1, 2, 3]
    else:
        df['new'] = pd.Series([1, 2, 3])

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text):
        es.update_dataframe(entity_id='customers', df=df.drop(columns=['cohort']))

    error_text = 'Updated dataframe contains 16 columns, expecting 15'
    with pytest.raises(ValueError, match=error_text):
        es.update_dataframe(entity_id='customers', df=df)

    # test already_sorted on entity without time index
    df = es["sessions"].df.copy()
    updated_id = to_pandas(df['id'])
    updated_id.iloc[1] = 2
    updated_id.iloc[2] = 1

    if es.dataframe_type == Library.KOALAS.value:
        df["id"] = updated_id.to_list()
        df = df.sort_index()
    else:
        df["id"] = updated_id
    es.update_dataframe(entity_id='sessions', df=df.copy())
    sessions_df = to_pandas(es['sessions'].df)
    assert sessions_df["id"].iloc[1] == 2  # no sorting since time index not defined
    es.update_dataframe(entity_id='sessions', df=df.copy(), already_sorted=True)
    sessions_df = to_pandas(es['sessions'].df)
    assert sessions_df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = es["customers"].df.copy()
    updated_signup = to_pandas(df['signup_date'])
    updated_signup.iloc[0] = datetime(2011, 4, 11)

    if es.dataframe_type == Library.KOALAS.value:
        df['signup_date'] = updated_signup.to_list()
        df = df.sort_index()
    else:
        df['signup_date'] = updated_signup
    es.update_dataframe(entity_id='customers', df=df.copy(), already_sorted=True)
    customers_df = to_pandas(es['customers'].df)
    assert customers_df["id"].iloc[0] == 2

    # only pandas allows for sorting:
    if isinstance(df, pd.DataFrame):
        es.update_dataframe(entity_id='customers', df=df.copy())
        assert es['customers'].df["id"].iloc[0] == 0
