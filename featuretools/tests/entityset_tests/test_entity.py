from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import Entity, EntitySet
from featuretools.tests.testing_utils import (
    make_ecommerce_entityset,
    to_pandas
)
from featuretools.utils.gen_utils import import_or_none
from featuretools.variable_types import find_variable_types

ks = import_or_none('databricks.koalas')


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
    other_es['log'].add_interesting_values()
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


def test_update_data(es):
    df = es['customers'].df.copy()
    if ks and isinstance(df, ks.DataFrame):
        df['new'] = [1, 2, 3]
    else:
        df['new'] = pd.Series([1, 2, 3])

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text):
        es['customers'].update_data(df.drop(columns=['cohort']))

    error_text = 'Updated dataframe contains 16 columns, expecting 15'
    with pytest.raises(ValueError, match=error_text):
        es['customers'].update_data(df)

    # test already_sorted on entity without time index
    df = es["sessions"].df.copy()
    updated_id = to_pandas(df['id'])
    updated_id.iloc[1] = 2
    updated_id.iloc[2] = 1

    if ks and isinstance(df, ks.DataFrame):
        df["id"] = updated_id.to_list()
        df = df.sort_index()
    else:
        df["id"] = updated_id
    es["sessions"].update_data(df.copy())
    sessions_df = to_pandas(es['sessions'].df)
    assert sessions_df["id"].iloc[1] == 2  # no sorting since time index not defined
    es["sessions"].update_data(df.copy(), already_sorted=True)
    sessions_df = to_pandas(es['sessions'].df)
    assert sessions_df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = es["customers"].df.copy()
    updated_signup = to_pandas(df['signup_date'])
    updated_signup.iloc[0] = datetime(2011, 4, 11)

    if ks and isinstance(df, ks.DataFrame):
        df['signup_date'] = updated_signup.to_list()
        df = df.sort_index()
    else:
        df['signup_date'] = updated_signup
    es["customers"].update_data(df.copy(), already_sorted=True)
    customers_df = to_pandas(es['customers'].df)
    assert customers_df["id"].iloc[0] == 2

    # only pandas allows for sorting:
    if isinstance(df, pd.DataFrame):
        es["customers"].update_data(df.copy())
        assert es['customers'].df["id"].iloc[0] == 0


def test_delete_variables(es):
    entity = es['customers']
    to_delete = ['age', 'cohort', 'email']
    entity.delete_variables(to_delete)

    variable_names = [v.id for v in entity.variables]

    for var in to_delete:
        assert var not in variable_names
        assert var not in entity.df.columns


def test_delete_variables_string_input(es):
    entity = es['customers']
    with pytest.raises(TypeError, match='variable_ids must be a list of variable names'):
        entity.delete_variables('age')

    variable_names = [v.id for v in entity.variables]

    assert 'age' in variable_names
    assert 'age' in entity.df.columns


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


def test_replace_latlong_nan_during_entity_creation(pd_es):
    nan_es = ft.EntitySet("latlong_nan")
    df = pd_es['log'].df.copy()
    df['latlong'][0] = np.nan

    with pytest.warns(UserWarning, match="LatLong columns should contain only tuples. All single 'NaN' values in column 'latlong' have been replaced with '\\(NaN, NaN\\)'."):
        entity = ft.Entity(id="nan_latlong_entity", df=df, entityset=nan_es, variable_types=pd_es['log'].variable_types)
    assert entity.df['latlong'][0] == (np.nan, np.nan)


def test_text_deprecation_warning():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": ["a", "c", "b", "a", "a"]
    })

    for text_repr in ['text', ft.variable_types.Text]:
        es = ft.EntitySet()
        match = "Text has been deprecated. Please use NaturalLanguage instead."
        with pytest.warns(FutureWarning, match=match):
            es = es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                          variable_types={"value": text_repr})

    for nl_repr in ['natural_language', ft.variable_types.NaturalLanguage]:
        es = ft.EntitySet()
        with pytest.warns(None) as record:
            es = es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                          variable_types={"value": nl_repr})
        assert len(record) == 0
