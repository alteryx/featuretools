from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

import featuretools as ft
from featuretools import variable_types
from featuretools.entityset import Entity, EntitySet
from featuretools.tests.testing_utils import make_ecommerce_entityset
from featuretools.variable_types import find_variable_types


def test_enforces_variable_id_is_str(es):
    assert variable_types.Categorical("1", es["customers"])

    error_text = 'Variable id must be a string'
    with pytest.raises(AssertionError, match=error_text):
        variable_types.Categorical(1, es["customers"])


def test_no_column_default_datetime(es):
    variable = variable_types.Datetime("new_time", es["customers"])
    assert variable.interesting_values.dtype == "datetime64[ns]"

    variable = variable_types.Timedelta("timedelta", es["customers"])
    assert variable.interesting_values.dtype == "timedelta64[ns]"


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
    other_es = make_ecommerce_entityset()
    latlong = es['log'].df['latlong'].copy()

    assert es['log'].__eq__(es['log'], deep=True)
    assert es['log'].__eq__(other_es['log'], deep=True)
    assert all(es['log'].df['latlong'].eq(latlong))

    other_es['log'].add_interesting_values()
    assert not es['log'].__eq__(other_es['log'], deep=True)

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
    df['new'] = pd.Series([1, 2, 3])

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text):
        es['customers'].update_data(df.drop(columns=['cohort']))

    error_text = 'Updated dataframe contains 16 columns, expecting 15'
    with pytest.raises(ValueError, match=error_text):
        es['customers'].update_data(df)

    # test already_sorted on entity without time index
    df = es["sessions"].df.copy()
    if isinstance(df, dd.DataFrame):
        updated_id = df['id'].compute()
    else:
        updated_id = df['id']
    updated_id.iloc[1:3] = [2, 1]
    df["id"] = updated_id
    es["sessions"].update_data(df.copy())
    sessions_df = es['sessions'].df
    if isinstance(sessions_df, dd.DataFrame):
        sessions_df = sessions_df.compute()
    assert sessions_df["id"].iloc[1] == 2  # no sorting since time index not defined
    es["sessions"].update_data(df.copy(), already_sorted=True)
    sessions_df = es['sessions'].df
    if isinstance(sessions_df, dd.DataFrame):
        sessions_df = sessions_df.compute()
    assert sessions_df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = es["customers"].df.copy()
    if isinstance(df, dd.DataFrame):
        updated_signup = df['signup_date'].compute()
    else:
        updated_signup = df['signup_date']
    updated_signup.iloc[0] = datetime(2011, 4, 11)
    df['signup_date'] = updated_signup
    es["customers"].update_data(df.copy(), already_sorted=True)
    customers_df = es['customers'].df
    if isinstance(customers_df, dd.DataFrame):
        customers_df = customers_df.compute()
    assert customers_df["id"].iloc[0] == 2

    # only pandas allows for sorting:
    if isinstance(df, pd.DataFrame):
        es["customers"].update_data(df.copy())
        assert es['customers'].df["id"].iloc[0] == 0


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


def test_query_by_values_secondary_time_index(es):
    end = np.datetime64(datetime(2011, 10, 1))
    all_instances = [0, 1, 2]
    result = es['customers'].query_by_values(all_instances, time_last=end)

    if isinstance(result, dd.DataFrame):
        result = result.compute().set_index('id')
    for col in ["cancel_date", "cancel_reason"]:
        nulls = result.loc[all_instances][col].isnull() == [False, True, True]
        assert nulls.all(), "Some instance has data it shouldn't for column %s" % col


def test_delete_variables(es):
    entity = es['customers']
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

    es = ft.EntitySet()
    variable_types = {'fraud': ft.variable_types.Boolean}
    old_variable_types = variable_types.copy()
    es.entity_from_dataframe(entity_id="transactions",
                             dataframe=df,
                             index='id',
                             time_index='transaction_time',
                             variable_types=variable_types)
    assert old_variable_types == variable_types


def test_passing_strings_to_variable_types_entity_init():
    variable_types = find_variable_types()
    reversed_variable_types = {str(v): k for k, v in variable_types.items()}
    reversed_variable_types['unknown variable'] = 'some unknown type string'

    es = EntitySet()
    dataframe = pd.DataFrame(columns=list(reversed_variable_types))
    with pytest.warns(UserWarning, match='Variable type {} was unrecognized, Unknown variable type was used instead'.format('some unknown type string')):
        entity = Entity('reversed_variable_types', dataframe, es,
                        variable_types=reversed_variable_types,
                        index="<class 'featuretools.variable_types.variable.Index'>",
                        time_index="<class 'featuretools.variable_types.variable.NumericTimeIndex'>",
                        )

    reversed_variable_types["unknown variable"] = "unknown"
    for variable in entity.variables:
        variable_class = variable.__class__
        assert variable_class.type_string == reversed_variable_types[variable.id]


def test_passing_strings_to_variable_types_from_dataframe():
    variable_types = find_variable_types()
    reversed_variable_types = {str(v): k for k, v in variable_types.items()}
    reversed_variable_types['unknown variable'] = 'some unknown type string'

    es = EntitySet()
    dataframe = pd.DataFrame(columns=list(reversed_variable_types))
    with pytest.warns(UserWarning, match='Variable type {} was unrecognized, Unknown variable type was used instead'.format('some unknown type string')):
        es.entity_from_dataframe(
            entity_id="reversed_variable_types",
            dataframe=dataframe,
            index="<class 'featuretools.variable_types.variable.Index'>",
            time_index="<class 'featuretools.variable_types.variable.NumericTimeIndex'>",
            variable_types=reversed_variable_types)

    entity = es["reversed_variable_types"]
    reversed_variable_types["unknown variable"] = "unknown"
    for variable in entity.variables:
        variable_class = variable.__class__
        assert variable_class.type_string == reversed_variable_types[variable.id]


def test_passing_strings_to_variable_types_dfs():
    variable_types = find_variable_types()
    teams = pd.DataFrame({
        'id': range(3),
        'name': ['Breakers', 'Spirit', 'Thorns']
    })
    games = pd.DataFrame({
        'id': range(5),
        'home_team_id': [2, 2, 1, 0, 1],
        'away_team_id': [1, 0, 2, 1, 0],
        'home_team_score': [3, 0, 1, 0, 4],
        'away_team_score': [2, 1, 2, 0, 0]
    })
    entities = {'teams': (teams, 'id', None, {'name': 'text'}), 'games': (games, 'id')}
    relationships = [('teams', 'id', 'games', 'home_team_id')]

    features = ft.dfs(entities, relationships, target_entity="teams", features_only=True)
    name_class = features[0].entity['name'].__class__
    assert name_class == variable_types['text']


def test_replace_latlong_nan_during_entity_creation(pd_es):
    nan_es = ft.EntitySet("latlong_nan")
    df = pd_es['log'].df.copy()
    df['latlong'][0] = np.nan

    with pytest.warns(UserWarning, match="LatLong columns should contain only tuples. All single 'NaN' values in column 'latlong' have been replaced with '\\(NaN, NaN\\)'."):
        entity = ft.Entity(id="nan_latlong_entity", df=df, entityset=nan_es, variable_types=pd_es['log'].variable_types)
    assert entity.df['latlong'][0] == (np.nan, np.nan)
