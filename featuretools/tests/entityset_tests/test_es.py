# -*- coding: utf-8 -*-

import copy
from builtins import range
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools import variable_types
from featuretools.entityset import EntitySet, Relationship


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_operations_invalidate_metadata(entityset):
    new_es = ft.EntitySet(id="test")
    # test metadata gets created on access
    assert new_es._data_description is None
    assert new_es.metadata is not None  # generated after access
    assert new_es._data_description is not None

    new_es.entity_from_dataframe("customers",
                                 entityset["customers"].df,
                                 index=entityset["customers"].index)
    new_es.entity_from_dataframe("sessions",
                                 entityset["sessions"].df,
                                 index=entityset["sessions"].index)
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    r = ft.Relationship(new_es["customers"]["id"],
                        new_es["sessions"]["customer_id"])
    new_es = new_es.add_relationship(r)
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es = new_es.normalize_entity("customers", "cohort", "cohort")
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es.add_last_time_indexes()
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es.add_interesting_values()
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None


def test_reset_metadata(entityset):
    assert entityset.metadata is not None
    assert entityset._data_description is not None
    entityset.reset_data_description()
    assert entityset._data_description is None


def test_cannot_readd_relationships_that_already_exists(entityset):
    before_len = len(entityset.relationships)
    entityset.add_relationship(entityset.relationships[0])
    after_len = len(entityset.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(entityset):
    for r in entityset.relationships:
        parent_e = entityset[r.parent_entity.id]
        child_e = entityset[r.child_entity.id]
        assert type(r.parent_variable) == variable_types.Index
        assert type(r.child_variable) == variable_types.Id
        assert parent_e.df[r.parent_variable.id].dtype == child_e.df[r.child_variable.id].dtype


def test_add_relationship_errors_on_dtype_mismatch(entityset):
    log_2_df = entityset['log'].df.copy()
    log_variable_types = {
        'id': variable_types.Categorical,
        'session_id': variable_types.Id,
        'product_id': variable_types.Id,
        'datetime': variable_types.Datetime,
        'value': variable_types.Numeric,
        'value_2': variable_types.Numeric,
        'latlong': variable_types.LatLong,
        'latlong2': variable_types.LatLong,
        'value_many_nans': variable_types.Numeric,
        'priority_level': variable_types.Ordinal,
        'purchased': variable_types.Boolean,
        'comments': variable_types.Text
    }
    entityset.entity_from_dataframe(entity_id='log2',
                                    dataframe=log_2_df,
                                    index='id',
                                    variable_types=log_variable_types,
                                    time_index='datetime')

    error_text = u'Unable to add relationship because id in customers is Pandas dtype category and session_id in log2 is Pandas dtype int64.'
    with pytest.raises(ValueError, match=error_text):
        mismatch = Relationship(entityset[u'customers']['id'], entityset['log2']['session_id'])
        entityset.add_relationship(mismatch)


def test_query_by_id(entityset):
    df = entityset['log'].query_by_values(instance_vals=[0])
    assert df['id'].values[0] == 0


def test_query_by_id_with_time(entityset):
    df = entityset['log'].query_by_values(
        instance_vals=[0, 1, 2, 3, 4],
        time_last=datetime(2011, 4, 9, 10, 30, 2 * 6))

    assert df['id'].get_values().tolist() == [0, 1, 2]


def test_get_forward_entities_deep(entityset):
    entities = entityset.get_forward_entities('log', 'deep')
    assert entities == set(['sessions', 'customers', 'products', u'régions', 'cohorts'])


def test_query_by_variable_with_time(entityset):
    df = entityset['log'].query_by_values(
        instance_vals=[0, 1, 2], variable_id='session_id',
        time_last=datetime(2011, 4, 9, 10, 50, 0))

    true_values = [
        i * 5 for i in range(5)] + [i * 1 for i in range(4)] + [0]
    assert df['id'].get_values().tolist() == list(range(10))
    assert df['value'].get_values().tolist() == true_values


def test_query_by_variable_with_training_window(entityset):
    df = entityset['log'].query_by_values(
        instance_vals=[0, 1, 2], variable_id='session_id',
        time_last=datetime(2011, 4, 9, 10, 50, 0),
        training_window='15m')

    assert df['id'].get_values().tolist() == [9]
    assert df['value'].get_values().tolist() == [0]


def test_query_by_indexed_variable(entityset):
    df = entityset['log'].query_by_values(
        instance_vals=['taco clock'],
        variable_id='product_id')

    assert df['id'].get_values().tolist() == [15, 16]


def test_check_variables_and_dataframe():
    # matches
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}
    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe('test_entity', df, index='id',
                                    variable_types=vtypes)
    assert entityset.entity_dict['test_entity'].variable_types['category'] == variable_types.Categorical


def test_make_index_variable_ordering():
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity',
                                    index='id1',
                                    make_index=True,
                                    variable_types=vtypes,
                                    dataframe=df)
    assert entityset.entity_dict['test_entity'].df.columns[0] == 'id1'


def test_extra_variable_type():
    # more variables
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical,
              'category2': variable_types.Categorical}

    error_text = "Variable ID category2 not in DataFrame"
    with pytest.raises(LookupError, match=error_text):
        entityset = EntitySet(id='test')
        entityset.entity_from_dataframe(entity_id='test_entity',
                                        index='id',
                                        variable_types=vtypes, dataframe=df)


def test_add_parent_not_index_varible(entityset):
    error_text = "Parent variable.*is not the index of entity Entity.*"
    with pytest.raises(AttributeError, match=error_text):
        entityset.add_relationship(Relationship(entityset[u'régions']['language'],
                                                entityset['customers'][u'région_id']))


def test_none_index():
    df = pd.DataFrame({'category': [1, 2, 3], 'category2': ['1', '2', '3']})
    vtypes = {'category': variable_types.Categorical, 'category2': variable_types.Categorical}

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity',
                                    dataframe=df,
                                    variable_types=vtypes)
    assert entityset['test_entity'].index == 'category'
    assert isinstance(entityset['test_entity']['category'], variable_types.Index)


def test_unknown_index():
    # more variables
    df = pd.DataFrame({'category': ['a', 'b', 'a']})
    vtypes = {'category': variable_types.Categorical}

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity',
                                    index='id',
                                    variable_types=vtypes, dataframe=df)
    assert entityset['test_entity'].index == 'id'
    assert entityset['test_entity'].df['id'].tolist() == list(range(3))


def test_doesnt_remake_index():
    # more variables
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})

    error_text = "Cannot make index: index variable already present"
    with pytest.raises(RuntimeError, match=error_text):
        entityset = EntitySet(id='test')
        entityset.entity_from_dataframe(entity_id='test_entity',
                                        index='id',
                                        make_index=True,
                                        dataframe=df)


def test_bad_time_index_variable():
    df = pd.DataFrame({'category': ['a', 'b', 'a']})

    error_text = "Time index not found in dataframe"
    with pytest.raises(LookupError, match=error_text):
        entityset = EntitySet(id='test')
        entityset.entity_from_dataframe(entity_id='test_entity',
                                        index="id",
                                        dataframe=df,
                                        time_index='time')


def test_converts_variable_types_on_init():
    df = pd.DataFrame({'id': [0, 1, 2],
                       'category': ['a', 'b', 'a'],
                       'category_int': [1, 2, 3],
                       'ints': ['1', '2', '3'],
                       'floats': ['1', '2', '3.0']})
    df["category_int"] = df["category_int"].astype("category")

    vtypes = {'id': variable_types.Categorical,
              'ints': variable_types.Numeric,
              'floats': variable_types.Numeric}
    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity', index='id',
                                    variable_types=vtypes, dataframe=df)

    entity_df = entityset['test_entity'].df
    assert entity_df['ints'].dtype.name in variable_types.PandasTypes._pandas_numerics
    assert entity_df['floats'].dtype.name in variable_types.PandasTypes._pandas_numerics

    # this is infer from pandas dtype
    e = entityset["test_entity"]
    assert isinstance(e['category_int'], variable_types.Categorical)


def test_converts_variable_type_after_init():
    df = pd.DataFrame({'id': [0, 1, 2],
                       'category': ['a', 'b', 'a'],
                       'ints': ['1', '2', '1']})

    df["category"] = df["category"].astype("category")

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity', index='id',
                                    dataframe=df)
    e = entityset['test_entity']
    df = entityset['test_entity'].df

    e.convert_variable_type('ints', variable_types.Numeric)
    assert isinstance(e['ints'], variable_types.Numeric)
    assert df['ints'].dtype.name in variable_types.PandasTypes._pandas_numerics

    e.convert_variable_type('ints', variable_types.Categorical)
    assert isinstance(e['ints'], variable_types.Categorical)

    e.convert_variable_type('ints', variable_types.Ordinal)
    assert isinstance(e['ints'], variable_types.Ordinal)

    e.convert_variable_type('ints', variable_types.Boolean,
                            true_val=1, false_val=2)
    assert isinstance(e['ints'], variable_types.Boolean)
    assert df['ints'].dtype.name == 'bool'


def test_converts_datetime():
    # string converts to datetime correctly
    # This test fails without defining vtypes.  Entityset
    # infers time column should be numeric type
    times = pd.date_range('1/1/2011', periods=3, freq='H')
    time_strs = times.strftime('%Y-%m-%d')
    df = pd.DataFrame({'id': [0, 1, 2], 'time': time_strs})
    vtypes = {'id': variable_types.Categorical,
              'time': variable_types.Datetime}

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(
        entity_id='test_entity',
        index='id',
        time_index="time",
        variable_types=vtypes,
        dataframe=df)
    pd_col = entityset['test_entity'].df['time']
    # assert type(entityset['test_entity']['time']) == variable_types.Datetime
    assert type(pd_col[0]) == pd.Timestamp


def test_handles_datetime_format():
    # check if we load according to the format string
    # pass in an ambigious date
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp('Jan 2, 2011')
    time_strs = [actual.strftime(datetime_format)] * 3
    df = pd.DataFrame(
        {'id': [0, 1, 2], 'time_format': time_strs, 'time_no_format': time_strs})
    vtypes = {'id': variable_types.Categorical,
              'time_format': (variable_types.Datetime, {"format": datetime_format}),
              'time_no_format': variable_types.Datetime}

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(
        entity_id='test_entity',
        index='id',
        variable_types=vtypes,
        dataframe=df)

    col_format = entityset['test_entity'].df['time_format']
    col_no_format = entityset['test_entity'].df['time_no_format']
    # without formatting pandas gets it wrong
    assert (col_no_format != actual).all()

    # with formatting we correctly get jan2
    assert (col_format == actual).all()


def test_handles_datetime_mismatch():
    # can't convert arbitrary strings
    df = pd.DataFrame({'id': [0, 1, 2], 'time': ['a', 'b', 'tomorrow']})
    vtypes = {'id': variable_types.Categorical,
              'time': variable_types.Datetime}

    error_text = "Given date string not likely a datetime."
    with pytest.raises(ValueError, match=error_text):
        entityset = EntitySet(id='test')
        entityset.entity_from_dataframe('test_entity', df, 'id',
                                        time_index='time', variable_types=vtypes)


def test_entity_init(entityset):
    # Note: to convert the time column directly either the variable type
    # or convert_date_columns must be specifie
    df = pd.DataFrame({'id': [0, 1, 2],
                       'time': [datetime(2011, 4, 9, 10, 31, 3 * i)
                                for i in range(3)],
                       'category': ['a', 'b', 'a'],
                       'number': [4, 5, 6]})

    vtypes = {'time': variable_types.Datetime}
    entityset.entity_from_dataframe('test_entity', df, index='id',
                                    time_index='time', variable_types=vtypes)
    assert entityset['test_entity'].df.shape == df.shape
    assert entityset['test_entity'].index == 'id'
    assert entityset['test_entity'].time_index == 'time'
    assert set([v.id for v in entityset['test_entity'].variables]) == set(df.columns)

    assert entityset['test_entity'].df['time'].dtype == df['time'].dtype
    assert set(entityset['test_entity'].df['id']) == set(df['id'])


def test_nonstr_column_names():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 3: ['a', 'b', 'c']})
    es = ft.EntitySet(id='Failure')

    error_text = "All column names must be strings.*"
    with pytest.raises(ValueError, match=error_text) as excinfo:
        es.entity_from_dataframe(entity_id='str_cols',
                                 dataframe=df,
                                 index='index')
    assert 'All column names must be strings (Column 3 is not a string)' in str(excinfo)


def test_sort_time_id():
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s")[::-1]})

    es = EntitySet("test", entities={"t": (
        transactions_df, "id", "transaction_time")})
    times = es["t"].df.transaction_time.tolist()
    assert times == sorted(transactions_df.transaction_time.tolist())


def test_already_sorted_parameter():
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "transaction_time": [datetime(2014, 4, 6),
                                                         datetime(
                                                             2012, 4, 8),
                                                         datetime(
                                                             2012, 4, 8),
                                                         datetime(
                                                             2013, 4, 8),
                                                         datetime(
                                                             2015, 4, 8),
                                                         datetime(2016, 4, 9)]})

    es = EntitySet(id='test')
    es.entity_from_dataframe('t',
                             transactions_df,
                             index='id',
                             time_index="transaction_time",
                             already_sorted=True)
    times = es["t"].df.transaction_time.tolist()
    assert times == transactions_df.transaction_time.tolist()


def test_concat_entitysets(entityset):
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}
    entityset.entity_from_dataframe(entity_id='test_entity',
                                    index='id1',
                                    make_index=True,
                                    variable_types=vtypes,
                                    dataframe=df)
    entityset.add_last_time_indexes()

    assert entityset.__eq__(entityset)
    entityset_1 = copy.deepcopy(entityset)
    entityset_2 = copy.deepcopy(entityset)

    emap = {
        'log': [list(range(10)) + [14, 15, 16], list(range(10, 14)) + [15, 16]],
        'sessions': [[0, 1, 2, 5], [1, 3, 4, 5]],
        'customers': [[0, 2], [1, 2]],
        'test_entity': [[0, 1], [0, 2]],
    }

    assert entityset.__eq__(entityset_1, deep=True)
    assert entityset.__eq__(entityset_2, deep=True)

    for i, es in enumerate([entityset_1, entityset_2]):
        for entity, rows in emap.items():
            df = es[entity].df
            es[entity].update_data(df=df.loc[rows[i]])

    assert 10 not in entityset_1['log'].last_time_index.index
    assert 10 in entityset_2['log'].last_time_index.index
    assert 9 in entityset_1['log'].last_time_index.index
    assert 9 not in entityset_2['log'].last_time_index.index
    assert not entityset.__eq__(entityset_1, deep=True)
    assert not entityset.__eq__(entityset_2, deep=True)

    # make sure internal indexes work before concat
    regions = entityset_1['customers'].query_by_values(['United States'], variable_id=u'région_id')
    assert regions.index.isin(entityset_1['customers'].df.index).all()

    assert entityset_1.__eq__(entityset_2)
    assert not entityset_1.__eq__(entityset_2, deep=True)

    old_entityset_1 = copy.deepcopy(entityset_1)
    old_entityset_2 = copy.deepcopy(entityset_2)
    entityset_3 = entityset_1.concat(entityset_2)

    assert old_entityset_1.__eq__(entityset_1, deep=True)
    assert old_entityset_2.__eq__(entityset_2, deep=True)

    assert entityset_3.__eq__(entityset, deep=True)
    for entity in entityset.entities:
        df = entityset[entity.id].df.sort_index()
        df_3 = entityset_3[entity.id].df.sort_index()
        for column in df:
            for x, y in zip(df[column], df_3[column]):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
        orig_lti = entityset[entity.id].last_time_index.sort_index()
        new_lti = entityset_3[entity.id].last_time_index.sort_index()
        for x, y in zip(orig_lti, new_lti):
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

    entityset_1['stores'].last_time_index = None
    entityset_1['test_entity'].last_time_index = None
    entityset_2['test_entity'].last_time_index = None
    entityset_4 = entityset_1.concat(entityset_2)
    assert not entityset_4.__eq__(entityset, deep=True)
    for entity in entityset.entities:
        df = entityset[entity.id].df.sort_index()
        df_4 = entityset_4[entity.id].df.sort_index()
        for column in df:
            for x, y in zip(df[column], df_4[column]):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

        if entity.id != 'test_entity':
            orig_lti = entityset[entity.id].last_time_index.sort_index()
            new_lti = entityset_4[entity.id].last_time_index.sort_index()
            for x, y in zip(orig_lti, new_lti):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
        else:
            assert entityset_4[entity.id].last_time_index is None


def test_set_time_type_on_init():
    # create cards entity
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    entityset = EntitySet("fraud", entities, relationships)
    # assert time_type is set
    assert entityset.time_type == variable_types.NumericTimeIndex


def test_sets_time_when_adding_entity():
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    accounts_df = pd.DataFrame({"id": [3, 4, 5],
                                "signup_date": [datetime(2002, 5, 1),
                                                datetime(2006, 3, 20),
                                                datetime(2011, 11, 11)]})
    accounts_df_string = pd.DataFrame({"id": [3, 4, 5],
                                       "signup_date": ["element",
                                                       "exporting",
                                                       "editable"]})
    # create empty entityset
    entityset = EntitySet("fraud")
    # assert it's not set
    assert getattr(entityset, "time_type", None) is None
    # add entity
    entityset.entity_from_dataframe("transactions",
                                    transactions_df,
                                    index="id",
                                    time_index="transaction_time")
    # assert time_type is set
    assert entityset.time_type == variable_types.NumericTimeIndex
    # add another entity
    entityset.normalize_entity("transactions",
                               "cards",
                               "card_id",
                               make_time_index=True)
    # assert time_type unchanged
    assert entityset.time_type == variable_types.NumericTimeIndex
    # add wrong time type entity
    error_text = "accounts time index is <class 'featuretools.variable_types.variable.DatetimeTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        entityset.entity_from_dataframe("accounts",
                                        accounts_df,
                                        index="id",
                                        time_index="signup_date")
    # add non time type as time index
    error_text = "Attempted to convert all string column signup_date to numeric"
    with pytest.raises(TypeError, match=error_text):
        entityset.entity_from_dataframe("accounts",
                                        accounts_df_string,
                                        index="id",
                                        time_index="signup_date")


def test_checks_time_type_setting_time_index(entityset):
    # set non time type as time index
    error_text = 'log time index not recognized as numeric or datetime'
    with pytest.raises(TypeError, match=error_text):
        entityset['log'].set_time_index('purchased')


def test_checks_time_type_setting_secondary_time_index(entityset):
    # entityset is timestamp time type
    assert entityset.time_type == variable_types.DatetimeTimeIndex
    # add secondary index that is timestamp type
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'cancel_date': ['cancel_date', 'cancel_reason']}
    entityset["customers"].set_secondary_time_index(new_2nd_ti)
    assert entityset.time_type == variable_types.DatetimeTimeIndex
    # add secondary index that is numeric type
    new_2nd_ti = {'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is <class 'featuretools.variable_types.variable.NumericTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        entityset["customers"].set_secondary_time_index(new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {'favorite_quote': ['favorite_quote', 'loves_ice_cream']}

    error_text = "data type \"All members of the working classes must seize the means of production.\" not understood"
    with pytest.raises(TypeError, match=error_text):
        entityset["customers"].set_secondary_time_index(new_2nd_ti)
    # add mismatched pair of secondary time indexes
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is <class 'featuretools.variable_types.variable.NumericTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        entityset["customers"].set_secondary_time_index(new_2nd_ti)

    # create entityset with numeric time type
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "card_id": [1, 2, 1, 3, 4, 5],
        "transaction_time": [10, 12, 13, 20, 21, 20],
        "fraud_decision_time": [11, 14, 15, 21, 22, 21],
        "transaction_city": ["City A"] * 6,
        "transaction_date": [datetime(1989, 2, i) for i in range(1, 7)],
        "fraud": [True, False, False, False, True, True]
    })
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    card_es = EntitySet("fraud", entities, relationships)
    assert card_es.time_type == variable_types.NumericTimeIndex
    # add secondary index that is numeric time type
    new_2nd_ti = {'fraud_decision_time': ['fraud_decision_time', 'fraud']}
    card_es['transactions'].set_secondary_time_index(new_2nd_ti)
    assert card_es.time_type == variable_types.NumericTimeIndex
    # add secondary index that is timestamp type
    new_2nd_ti = {'transaction_date': ['transaction_date', 'fraud']}

    error_text = "transactions time index is <class 'featuretools.variable_types.variable.DatetimeTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        card_es['transactions'].set_secondary_time_index(new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {'transaction_city': ['transaction_city', 'fraud']}

    error_text = 'data type \"City A\" not understood'
    with pytest.raises(TypeError, match=error_text):
        card_es['transactions'].set_secondary_time_index(new_2nd_ti)
    # add mixed secondary time indexes
    new_2nd_ti = {'transaction_city': ['transaction_city', 'fraud'],
                  'fraud_decision_time': ['fraud_decision_time', 'fraud']}
    with pytest.raises(TypeError, match=error_text):
        card_es['transactions'].set_secondary_time_index(new_2nd_ti)

    # add bool secondary time index
    error_text = 'transactions time index not recognized as numeric or datetime'
    with pytest.raises(TypeError, match=error_text):
        card_es['transactions'].set_secondary_time_index({'fraud': ['fraud']})


def test_related_instances_backward(entityset):
    result = entityset.related_instances(
        start_entity_id=u'régions', final_entity_id='log',
        instance_ids=['United States'])

    col = entityset['log'].df['id'].values
    assert len(result['id'].values) == len(col)
    assert set(result['id'].values) == set(col)

    result = entityset.related_instances(
        start_entity_id=u'régions', final_entity_id='log',
        instance_ids=['Mexico'])

    assert len(result['id'].values) == 0


def test_related_instances_forward(entityset):
    result = entityset.related_instances(
        start_entity_id='log', final_entity_id=u'régions',
        instance_ids=[0, 1])

    assert len(result['id'].values) == 1
    assert result['id'].values[0] == 'United States'


def test_related_instances_mixed_path(entityset):
    result = entityset.related_instances(
        start_entity_id='customers', final_entity_id='products',
        instance_ids=[1])
    related = ["Haribo sugar-free gummy bears", "coke zero"]
    assert set(related) == set(result['id'].values)


def test_related_instances_all(entityset):
    # test querying across the entityset
    result = entityset.related_instances(
        start_entity_id='customers', final_entity_id='products',
        instance_ids=None)

    for p in entityset['products'].df['id'].values:
        assert p in result['id'].values


def test_related_instances_all_cutoff_time_same_entity(entityset):
    # test querying across the entityset
    result = entityset.related_instances(
        start_entity_id='log', final_entity_id='log',
        instance_ids=None, time_last=pd.Timestamp('2011/04/09 10:30:31'))

    assert result['id'].values.tolist() == list(range(5))


def test_get_pandas_slice(entityset):
    filter_eids = ['products', u'régions', 'customers']
    result = entityset.get_pandas_data_slice(filter_entity_ids=filter_eids,
                                             index_eid='customers',
                                             instances=[0])

    # make sure all necessary filter frames are present
    assert set(result.keys()) == set(filter_eids)
    assert set(result['products'].keys()), set(['products', 'log'])
    assert set(result['customers'].keys()) == set(
        ['customers', 'sessions', 'log'])
    assert set(result[u'régions'].keys()) == set(
        [u'régions', 'stores', 'customers', 'sessions', 'log'])

    # make sure different subsets of the log are included in each filtering
    assert set(result['customers']['log']['id'].values) == set(range(10))
    assert set(result['products']['log']['id'].values) == set(
        list(range(10)) + list(range(11, 15)))
    assert set(result[u'régions']['log']['id'].values) == set(range(17))


def test_get_pandas_slice_times(entityset):
    # todo these test used to use time first time last. i remvoed and it
    # still passes,but we should double check this okay
    filter_eids = ['products', u'régions', 'customers']
    start = np.datetime64(datetime(2011, 4, 1))
    end = np.datetime64(datetime(2011, 4, 9, 10, 31, 10))
    result = entityset.get_pandas_data_slice(filter_entity_ids=filter_eids,
                                             index_eid='customers',
                                             instances=[0],
                                             time_last=end)

    # make sure no times outside range are included in any frames
    for eid in filter_eids:
        for t in result[eid]['log']['datetime'].values:
            assert t >= start and t < end

        # the instance ids should be the same for all filters
        for i in range(7):
            assert i in result[eid]['log']['id'].values


def test_get_pandas_slice_times_include(entityset):
    # todo these test used to use time first time last. i remvoed and it
    # still passes,but we should double check this okay
    filter_eids = ['products', u'régions', 'customers']
    start = np.datetime64(datetime(2011, 4, 1))
    end = np.datetime64(datetime(2011, 4, 9, 10, 31, 10))
    result = entityset.get_pandas_data_slice(filter_entity_ids=filter_eids,
                                             index_eid='customers',
                                             instances=[0],
                                             time_last=end)

    # make sure no times outside range are included in any frames
    for eid in filter_eids:
        for t in result[eid]['log']['datetime'].values:
            assert t >= start and t <= end

        # the instance ids should be the same for all filters
        for i in range(7):
            assert i in result[eid]['log']['id'].values


def test_get_pandas_slice_secondary_index(entityset):
    filter_eids = ['products', u'régions', 'customers']
    # this date is before the cancel date of customers 1 and 2
    end = np.datetime64(datetime(2011, 10, 1))
    all_instances = [0, 1, 2]
    result = entityset.get_pandas_data_slice(filter_entity_ids=filter_eids,
                                             index_eid='customers',
                                             instances=all_instances,
                                             time_last=end)

    # only customer 0 should have values from these columns
    customers_df = result["customers"]["customers"]
    for col in ["cancel_date", "cancel_reason"]:
        nulls = customers_df.iloc[all_instances][col].isnull() == [
            False, True, True]
        assert nulls.all(), "Some instance has data it shouldn't for column %s" % col


def test_add_link_vars(entityset):
    eframes = {e_id: entityset[e_id].df
               for e_id in ["log", "sessions", "customers", u"régions"]}

    entityset._add_multigenerational_link_vars(frames=eframes,
                                               start_entity_id=u'régions',
                                               end_entity_id='log')

    assert 'sessions.customer_id' in eframes['log'].columns
    assert u'sessions.customers.région_id' in eframes['log'].columns


def test_normalize_entity(entityset):
    error_text = "'additional_variables' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        entityset.normalize_entity('sessions', 'device_types', 'device_type',
                                   additional_variables='log')

    error_text = "'copy_variables' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        entityset.normalize_entity('sessions', 'device_types', 'device_type',
                                   copy_variables='log')

    entityset.normalize_entity('sessions', 'device_types', 'device_type',
                               additional_variables=['device_name'],
                               make_time_index=False)

    assert len(entityset.get_forward_relationships('sessions')) == 2
    assert entityset.get_forward_relationships(
        'sessions')[1].parent_entity.id == 'device_types'
    assert 'device_name' in entityset['device_types'].df.columns
    assert 'device_name' not in entityset['sessions'].df.columns
    assert 'device_type' in entityset['device_types'].df.columns


def test_normalize_time_index_from_none(entityset):
    entityset['customers'].time_index = None
    entityset.normalize_entity('customers', 'birthdays', 'date_of_birth', make_time_index='date_of_birth')
    assert entityset['birthdays'].time_index == 'date_of_birth'


def test_raise_error_if_dupicate_additional_variables_passed(entityset):
    error_text = "'additional_variables' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        entityset.normalize_entity('sessions', 'device_types', 'device_type',
                                   additional_variables=['device_name', 'device_name'])


def test_raise_error_if_dupicate_copy_variables_passed(entityset):
    error_text = "'copy_variables' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        entityset.normalize_entity('sessions', 'device_types', 'device_type',
                                   copy_variables=['device_name', 'device_name'])


def test_normalize_entity_copies_variable_types(entityset):
    entityset['log'].convert_variable_type(
        'value', variable_types.Ordinal, convert_data=False)
    assert entityset['log'].variable_types['value'] == variable_types.Ordinal
    assert entityset['log'].variable_types['priority_level'] == variable_types.Ordinal
    entityset.normalize_entity('log', 'values_2', 'value_2',
                               additional_variables=['priority_level'],
                               copy_variables=['value'],
                               make_time_index=False)

    assert len(entityset.get_forward_relationships('log')) == 3
    assert entityset.get_forward_relationships(
        'log')[2].parent_entity.id == 'values_2'
    assert 'priority_level' in entityset['values_2'].df.columns
    assert 'value' in entityset['values_2'].df.columns
    assert 'priority_level' not in entityset['log'].df.columns
    assert 'value' in entityset['log'].df.columns
    assert 'value_2' in entityset['values_2'].df.columns
    assert entityset['values_2'].variable_types['priority_level'] == variable_types.Ordinal
    assert entityset['values_2'].variable_types['value'] == variable_types.Ordinal


def test_make_time_index_keeps_original_sorting():
    trips = {
        'trip_id': [999 - i for i in range(1000)],
        'flight_time': [datetime(1997, 4, 1) for i in range(1000)],
        'flight_id': [1 for i in range(350)] + [2 for i in range(650)]
    }
    order = [i for i in range(1000)]
    df = pd.DataFrame.from_dict(trips)
    es = EntitySet('flights')
    es.entity_from_dataframe("trips",
                             dataframe=df,
                             index="trip_id",
                             time_index='flight_time')
    assert (es['trips'].df['trip_id'] == order).all()
    es.normalize_entity(base_entity_id="trips",
                        new_entity_id="flights",
                        index="flight_id",
                        make_time_index=True)
    assert (es['trips'].df['trip_id'] == order).all()


def test_normalize_entity_new_time_index(entityset):
    entityset.normalize_entity('log', 'values', 'value',
                               make_time_index=True,
                               new_entity_time_index="value_time")

    assert entityset['values'].time_index == 'value_time'
    assert 'value_time' in entityset['values'].df.columns
    assert len(entityset['values'].df.columns) == 2


def test_secondary_time_index(entityset):
    entityset.normalize_entity('log', 'values', 'value',
                               make_time_index=True,
                               make_secondary_time_index={
                                   'datetime': ['comments']},
                               new_entity_time_index="value_time",
                               new_entity_secondary_time_index='second_ti')

    assert (isinstance(entityset['values'].df['second_ti'], pd.Series))
    assert (entityset['values']['second_ti'].type_string == 'datetime')
    assert (entityset['values'].secondary_time_index == {
            'second_ti': ['comments', 'second_ti']})


def test_sizeof(entityset):
    total_size = 0
    for entity in entityset.entities:
        total_size += entity.df.__sizeof__()
        total_size += entity.last_time_index.__sizeof__()

    assert entityset.__sizeof__() == total_size


def test_construct_without_id():
    assert ft.EntitySet().id is None


def test_repr_without_id():
    match = 'Entityset: None\n  Entities:\n  Relationships:\n    No relationships'
    assert repr(ft.EntitySet()) == match


def test_getitem_without_id():
    error_text = 'Entity test does not exist'
    with pytest.raises(KeyError, match=error_text):
        ft.EntitySet()['test']


def test_metadata_without_id():
    es = ft.EntitySet()
    assert es.metadata.id is None


def test_datetime64_conversion():
    df = pd.DataFrame({'id': [0, 1, 2],
                       'ints': ['1', '2', '1']})
    df["time"] = pd.Timestamp.now()
    df["time"] = df["time"].astype("datetime64[ns, UTC]")

    entityset = EntitySet(id='test')
    entityset.entity_from_dataframe(entity_id='test_entity', index='id',
                                    dataframe=df)
    vtype_time_index = variable_types.variable.DatetimeTimeIndex
    entityset['test_entity'].convert_variable_type('time', vtype_time_index)
