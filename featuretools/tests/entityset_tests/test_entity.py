# -*- coding: utf-8 -*-
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools import variable_types


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_enforces_variable_id_is_str(es):
    assert variable_types.Categorical("1", es["customers"])

    error_text = 'Variable id must be a string'
    with pytest.raises(AssertionError, match=error_text):
        variable_types.Categorical(1, es["customers"])


def test_is_index_column(es):
    assert es['cohorts'].index == 'cohort'


def test_reorders_index():
    es = ft.EntitySet('test')
    df = pd.DataFrame({'id': [1, 2, 3], 'other': [4, 5, 6]})
    df.columns = ['other', 'id']
    es.entity_from_dataframe('test',
                             df,
                             index='id')
    assert es['test'].variables[0].id == 'id'
    assert es['test'].variables[0].id == es['test'].index
    assert [v.id for v in es['test'].variables] == list(es['test'].df.columns)


def test_index_at_beginning(es):
    for e in es.entity_dict.values():
        assert e.index == e.variables[0].id


def test_variable_ordering_matches_column_ordering(es):
    for e in es.entity_dict.values():
        assert [v.id for v in e.variables] == list(e.df.columns)


def test_eq(es):

    es['log'].id = 'customers'
    es['log'].index = 'notid'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].index = 'id'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].time_index = 'signup_date'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].secondary_time_index = {
        'cancel_date': ['cancel_reason', 'cancel_date']}
    assert not es['customers'].__eq__(es['log'], deep=True)


def test_update_data(es):
    df = es['customers'].df.copy()
    df['new'] = [1, 2, 3]

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text) as excinfo:
        es['customers'].update_data(df.drop(columns=['cohort']))
    assert 'Updated dataframe is missing new cohort column' in str(excinfo)

    error_text = 'Updated dataframe contains 15 columns, expecting 14'
    with pytest.raises(ValueError, match=error_text) as excinfo:
        es['customers'].update_data(df)
    assert 'Updated dataframe contains 15 columns, expecting 14' in str(excinfo)

    # test already_sorted on entity without time index
    df = es["sessions"].df.copy(deep=True)
    df["id"].iloc[1:3] = [2, 1]
    es["sessions"].update_data(df.copy(deep=True))
    assert es["sessions"].df["id"].iloc[1] == 1
    es["sessions"].update_data(df.copy(deep=True), already_sorted=True)
    assert es["sessions"].df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = es["customers"].df.copy(deep=True)
    df["signup_date"].iloc[0] = datetime(2011, 4, 11)
    es["customers"].update_data(df.copy(deep=True))
    assert es["customers"].df["id"].iloc[0] == 0
    es["customers"].update_data(df.copy(deep=True), already_sorted=True)
    assert es["customers"].df["id"].iloc[0] == 2


def test_query_by_values_returns_rows_in_given_order():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": ["a", "c", "b", "a", "a"],
        "time": [1000, 2000, 3000, 4000, 5000]
    })

    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                  time_index="time", variable_types={
                                      "value": ft.variable_types.Categorical
                                  })
    query = es['test'].query_by_values(['b', 'a'], variable_id='value')
    assert np.array_equal(query['id'], [1, 3, 4, 5])
