from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools import variable_types
from featuretools.tests.testing_utils import make_ecommerce_entityset


def test_enforces_variable_id_is_str(pd_es):
    assert variable_types.Categorical("1", pd_es["customers"])

    error_text = 'Variable id must be a string'
    with pytest.raises(AssertionError, match=error_text):
        variable_types.Categorical(1, pd_es["customers"])


def test_no_column_default_datetime(pd_es):
    variable = variable_types.Datetime("new_time", pd_es["customers"])
    assert variable.interesting_values.dtype == "datetime64[ns]"

    variable = variable_types.Timedelta("timedelta", pd_es["customers"])
    assert variable.interesting_values.dtype == "timedelta64[ns]"


def test_is_index_column(pd_es):
    assert pd_es['cohorts'].index == 'cohort'


def test_reorders_index():
    pd_es = ft.EntitySet('test')
    df = pd.DataFrame({'id': [1, 2, 3], 'other': [4, 5, 6]})
    df.columns = ['other', 'id']
    pd_es.entity_from_dataframe('test',
                                df,
                                index='id')
    assert pd_es['test'].variables[0].id == 'id'
    assert pd_es['test'].variables[0].id == pd_es['test'].index
    assert [v.id for v in pd_es['test'].variables] == list(pd_es['test'].df.columns)


def test_index_at_beginning(pd_es):
    for e in pd_es.entity_dict.values():
        assert e.index == e.variables[0].id


def test_variable_ordering_matches_column_ordering(pd_es):
    for e in pd_es.entity_dict.values():
        assert [v.id for v in e.variables] == list(e.df.columns)


def test_eq(pd_es):
    other_es = make_ecommerce_entityset()
    latlong = pd_es['log'].df['latlong'].copy()

    assert pd_es['log'].__eq__(pd_es['log'], deep=True)
    assert pd_es['log'].__eq__(other_es['log'], deep=True)
    assert (pd_es['log'].df['latlong'] == latlong).all()

    other_es['log'].add_interesting_values()
    assert not pd_es['log'].__eq__(other_es['log'], deep=True)

    pd_es['log'].id = 'customers'
    pd_es['log'].index = 'notid'
    assert not pd_es['customers'].__eq__(pd_es['log'], deep=True)

    pd_es['log'].index = 'id'
    assert not pd_es['customers'].__eq__(pd_es['log'], deep=True)

    pd_es['log'].time_index = 'signup_date'
    assert not pd_es['customers'].__eq__(pd_es['log'], deep=True)

    pd_es['log'].secondary_time_index = {
        'cancel_date': ['cancel_reason', 'cancel_date']}
    assert not pd_es['customers'].__eq__(pd_es['log'], deep=True)


def test_update_data(pd_es):
    df = pd_es['customers'].df.copy()
    df['new'] = [1, 2, 3]

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text):
        pd_es['customers'].update_data(df.drop(columns=['cohort']))

    error_text = 'Updated dataframe contains 16 columns, expecting 15'
    with pytest.raises(ValueError, match=error_text):
        pd_es['customers'].update_data(df)

    # test already_sorted on entity without time index
    df = pd_es["sessions"].df.copy(deep=True)
    df["id"].iloc[1:3] = [2, 1]
    pd_es["sessions"].update_data(df.copy(deep=True))
    assert pd_es["sessions"].df["id"].iloc[1] == 2  # no sorting since time index not defined
    pd_es["sessions"].update_data(df.copy(deep=True), already_sorted=True)
    assert pd_es["sessions"].df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = pd_es["customers"].df.copy(deep=True)
    df["signup_date"].iloc[0] = datetime(2011, 4, 11)
    pd_es["customers"].update_data(df.copy(deep=True))
    assert pd_es["customers"].df["id"].iloc[0] == 0
    pd_es["customers"].update_data(df.copy(deep=True), already_sorted=True)
    assert pd_es["customers"].df["id"].iloc[0] == 2


def test_query_by_values_returns_rows_in_given_order():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": ["a", "c", "b", "a", "a"],
        "time": [1000, 2000, 3000, 4000, 5000]
    })

    pd_es = ft.EntitySet()
    pd_es = pd_es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                        time_index="time", variable_types={
                                            "value": ft.variable_types.Categorical
                                        })
    query = pd_es['test'].query_by_values(['b', 'a'], variable_id='value')
    assert np.array_equal(query['id'], [1, 3, 4, 5])


def test_query_by_values_secondary_time_index(pd_es):
    end = np.datetime64(datetime(2011, 10, 1))
    all_instances = [0, 1, 2]
    result = pd_es['customers'].query_by_values(all_instances, time_last=end)

    for col in ["cancel_date", "cancel_reason"]:
        nulls = result.loc[all_instances][col].isnull() == [False, True, True]
        assert nulls.all(), "Some instance has data it shouldn't for column %s" % col


def test_delete_variables(pd_es):
    entity = pd_es['customers']
    to_delete = ['age', 'cohort', 'email']
    entity.delete_variables(to_delete)

    variable_names = [v.id for v in entity.variables]

    for var in to_delete:
        assert var not in variable_names
        assert var not in entity.df.columns


def test_variable_types_unmodified():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                       "transaction_time": [10, 12, 13, 20, 21, 20],
                       "fraud": [True, False, False, False, True, True]})

    pd_es = ft.EntitySet()
    variable_types = {'fraud': ft.variable_types.Boolean}
    old_variable_types = variable_types.copy()
    pd_es.entity_from_dataframe(entity_id="transactions",
                                dataframe=df,
                                index='id',
                                time_index='transaction_time',
                                variable_types=variable_types)
    assert old_variable_types == variable_types
