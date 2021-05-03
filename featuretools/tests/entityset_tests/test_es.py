# import copy
# import logging
import re
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import woodwork.logical_types as ltypes
import woodwork as ww

import featuretools as ft
from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean

ks = import_or_none('databricks.koalas')


def test_normalize_time_index_as_additional_variable(es):
    error_text = "Not moving signup_date as it is the base time index variable."
    with pytest.raises(ValueError, match=error_text):
        assert "signup_date" in es["customers"].columns
        es.normalize_dataframe(base_dataframe_id='customers',
                               new_dataframe_id='cancellations',
                               index='cancel_reason',
                               make_time_index='signup_date',
                               additional_columns=['signup_date'],
                               copy_columns=[])


def test_cannot_re_add_relationships_that_already_exists(es):
    warn_text = "Not adding duplicate relationship: " + str(es.relationships[0])
    before_len = len(es.relationships)
    rel = es.relationships[0]
    with pytest.warns(UserWarning, match=warn_text):
        es.add_relationship(relationship=rel)
    with pytest.warns(UserWarning, match=warn_text):
        es.add_relationship(rel._parent_dataframe_id, rel._parent_column_id,
                            rel._child_dataframe_id, rel._child_column_id)
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        parent_df = es[r.parent_dataframe.ww.name]
        child_df = es[r.child_dataframe.ww.name]
        assert parent_df.ww.index == r.parent_column.name
        assert 'foreign_key' in r.child_column.ww.semantic_tags
        assert str(parent_df[r.parent_column.name].dtype) == str(child_df[r.child_column.name].dtype)


def test_add_relationship_instantiated_logical_types(es):
    log_2_df = es['log'].copy()
    log_logical_types = {
        'id': ltypes.Integer,
        'session_id': ltypes.Integer,
        'product_id': ltypes.Categorical(),
        'datetime': ltypes.Datetime,
        'value': ltypes.Double,
        'value_2': ltypes.Double,
        'latlong': ltypes.LatLong,
        'latlong2': ltypes.LatLong,
        'zipcode': ltypes.PostalCode,
        'countrycode': ltypes.CountryCode,
        'subregioncode': ltypes.SubRegionCode,
        'value_many_nans': ltypes.Double,
        'priority_level': ltypes.Ordinal(order=[0, 1, 2]),
        'purchased': ltypes.Boolean,
        'comments': ltypes.NaturalLanguage
    }
    log_semantic_tags = {
        'session_id': 'foreign_key',
        'product_id': 'foreign_key'
    }
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(dataframe_id='log2',
                     dataframe=log_2_df,
                     index='id',
                     logical_types=log_logical_types,
                     semantic_tags=log_semantic_tags,
                     time_index='datetime')
    assert 'log2' in es.dataframe_dict
    assert es['log2'].ww.schema is not None
    assert es['log2'].ww.logical_types['product_id'] == ltypes.Categorical()
    assert es['products'].ww.logical_types['id'] == ltypes.Categorical

    category_dtype = 'category'
    if ks and isinstance(es['customers'], ks.DataFrame):
        category_dtype = 'string'

    # --> terrible error message
    warning_text = f'Logical type for child column Categorical does not match parent column logical type Categorical. Changing child logical type to match parent.'
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship(u'products', 'id', 'log2', 'product_id')
    assert es['log2'].ww.logical_types['product_id'] == ltypes.Categorical
    assert es['products'].ww.logical_types['id'] == ltypes.Categorical


def test_add_relationship_different_logical_types_same_dtype(es):
    log_2_df = es['log'].copy()
    log_logical_types = {
        'id': ltypes.Integer,
        'session_id': ltypes.Integer,
        'product_id': ltypes.CountryCode,
        'datetime': ltypes.Datetime,
        'value': ltypes.Double,
        'value_2': ltypes.Double,
        'latlong': ltypes.LatLong,
        'latlong2': ltypes.LatLong,
        'zipcode': ltypes.PostalCode,
        'countrycode': ltypes.CountryCode,
        'subregioncode': ltypes.SubRegionCode,
        'value_many_nans': ltypes.Double,
        'priority_level': ltypes.Ordinal(order=[0, 1, 2]),
        'purchased': ltypes.Boolean,
        'comments': ltypes.NaturalLanguage
    }
    log_semantic_tags = {
        'session_id': 'foreign_key',
        'product_id': 'foreign_key'
    }
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(dataframe_id='log2',
                     dataframe=log_2_df,
                     index='id',
                     logical_types=log_logical_types,
                     semantic_tags=log_semantic_tags,
                     time_index='datetime')
    assert 'log2' in es.dataframe_dict
    assert es['log2'].ww.schema is not None
    assert es['log2'].ww.logical_types['product_id'] == ltypes.CountryCode
    assert es['products'].ww.logical_types['id'] == ltypes.Categorical

    category_dtype = 'category'
    if ks and isinstance(es['customers'], ks.DataFrame):
        category_dtype = 'string'

    warning_text = f'Logical type for child column CountryCode does not match parent column logical type Categorical. Changing child logical type to match parent.'
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship(u'products', 'id', 'log2', 'product_id')
    assert es['log2'].ww.logical_types['product_id'] == ltypes.Categorical
    assert es['products'].ww.logical_types['id'] == ltypes.Categorical


def test_add_relationship_different_compatible_dtypes(es):
    log_2_df = es['log'].copy()
    log_logical_types = {
        'id': ltypes.Integer,
        'session_id': ltypes.Datetime,
        'product_id': ltypes.Categorical,
        'datetime': ltypes.Datetime,
        'value': ltypes.Double,
        'value_2': ltypes.Double,
        'latlong': ltypes.LatLong,
        'latlong2': ltypes.LatLong,
        'zipcode': ltypes.PostalCode,
        'countrycode': ltypes.CountryCode,
        'subregioncode': ltypes.SubRegionCode,
        'value_many_nans': ltypes.Double,
        'priority_level': ltypes.Ordinal(order=[0, 1, 2]),
        'purchased': ltypes.Boolean,
        'comments': ltypes.NaturalLanguage
    }
    log_semantic_tags = {
        'session_id': 'foreign_key',
        'product_id': 'foreign_key'
    }
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(dataframe_id='log2',
                     dataframe=log_2_df,
                     index='id',
                     logical_types=log_logical_types,
                     semantic_tags=log_semantic_tags,
                     time_index='datetime')
    assert 'log2' in es.dataframe_dict
    assert es['log2'].ww.schema is not None
    assert es['log2'].ww.logical_types['session_id'] == ltypes.Datetime
    assert es['customers'].ww.logical_types['id'] == ltypes.Integer

    category_dtype = 'category'
    if ks and isinstance(es['customers'], ks.DataFrame):
        category_dtype = 'string'

    warning_text = f'Logical type for child column Datetime does not match parent column logical type Integer. Changing child logical type to match parent.'
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship(u'customers', 'id', 'log2', 'session_id')
    assert es['log2'].ww.logical_types['session_id'] == ltypes.Integer
    assert es['customers'].ww.logical_types['id'] == ltypes.Integer


def test_add_relationship_errors_child_v_index(es):
    es.add_dataframe(dataframe_id='log2',
                     dataframe=es['log'].ww.copy())

    to_match = "Unable to add relationship because child column 'id' in 'log2' is also its index"
    with pytest.raises(ValueError, match=to_match):
        es.add_relationship('log', 'id', 'log2', 'id')


# def test_add_relationship_empty_child_convert_dtype(es):
#     relationship = ft.Relationship(es, "sessions", "id", "log", "session_id")
#     empty_log_df = pd.DataFrame(columns=es['log'].columns)
#     # --> schema isn't valid.... not sure what this is testing
#     empty_log_df.ww.init(schema=es['log'].ww.schema)

#     es.add_dataframe('log', empty_log_df)

#     assert len(es['log']) == 0
#     assert es['log']['session_id'].dtype == 'object'

#     es.relationships.remove(relationship)
#     assert(relationship not in es.relationships)

#     es.add_relationship(relationship=relationship)
#     assert es['log']['session_id'].dtype == 'int64'


def test_add_relationship_with_relationship_object(es):
    relationship = ft.Relationship(es, "sessions", "id", "log", "session_id")
    es.add_relationship(relationship=relationship)
    assert relationship in es.relationships


def test_add_relationships_with_relationship_object(es):
    relationships = [ft.Relationship(es, "sessions", "id", "log", "session_id")]
    es.add_relationships(relationships)
    assert relationships[0] in es.relationships


def test_add_relationship_error(es):
    relationship = ft.Relationship(es, "sessions", "id", "log", "session_id")
    error_message = "Cannot specify dataframe and column id values and also supply a Relationship"
    with pytest.raises(ValueError, match=error_message):
        es.add_relationship(parent_dataframe_id="sessions", relationship=relationship)

# --> needs to wait until query_by_values is implemented
# def test_query_by_values_returns_rows_in_given_order():
#     data = pd.DataFrame({
#         "id": [1, 2, 3, 4, 5],
#         "value": ["a", "c", "b", "a", "a"],
#         "time": [1000, 2000, 3000, 4000, 5000]
#     })

#     es = ft.EntitySet()
#     es = es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
#                                   time_index="time", variable_types={
#                                             "value": ft.variable_types.Categorical
#                                   })
#     query = es.query_by_values('test', ['b', 'a'], variable_id='value')
#     assert np.array_equal(query['id'], [1, 3, 4, 5])


# def test_query_by_values_secondary_time_index(es):
#     end = np.datetime64(datetime(2011, 10, 1))
#     all_instances = [0, 1, 2]
#     result = es.query_by_values('customers', all_instances, time_last=end)
#     result = to_pandas(result, index='id')

#     for col in ["cancel_date", "cancel_reason"]:
#         nulls = result.loc[all_instances][col].isnull() == [False, True, True]
#         assert nulls.all(), "Some instance has data it shouldn't for column %s" % col


# def test_query_by_id(es):
#     df = to_pandas(es.query_by_values('log', instance_vals=[0]))
#     assert df['id'].values[0] == 0


# def test_query_by_single_value(es):
#     df = to_pandas(es.query_by_values('log', instance_vals=0))
#     assert df['id'].values[0] == 0


# def test_query_by_df(es):
#     instance_df = pd.DataFrame({'id': [1, 3], 'vals': [0, 1]})
#     df = to_pandas(es.query_by_values('log', instance_vals=instance_df))

#     assert np.array_equal(df['id'], [1, 3])


# def test_query_by_id_with_time(es):
#     df = es.query_by_values(
#         entity_id='log',
#         instance_vals=[0, 1, 2, 3, 4],
#         time_last=datetime(2011, 4, 9, 10, 30, 2 * 6))
#     df = to_pandas(df)
#     if ks and isinstance(es['log'].df, ks.DataFrame):
#         # Koalas doesn't maintain order
#         df = df.sort_values('id')

#     assert list(df['id'].values) == [0, 1, 2]


# def test_query_by_variable_with_time(es):
#     df = es.query_by_values(
#         entity_id='log',
#         instance_vals=[0, 1, 2], variable_id='session_id',
#         time_last=datetime(2011, 4, 9, 10, 50, 0))
#     df = to_pandas(df)

#     true_values = [
#         i * 5 for i in range(5)] + [i * 1 for i in range(4)] + [0]
#     if ks and isinstance(es['log'].df, ks.DataFrame):
#         # Koalas doesn't maintain order
#         df = df.sort_values('id')

#     assert list(df['id'].values) == list(range(10))
#     assert list(df['value'].values) == true_values


# def test_query_by_variable_with_training_window(es):
#     df = es.query_by_values(
#         entity_id='log',
#         instance_vals=[0, 1, 2], variable_id='session_id',
#         time_last=datetime(2011, 4, 9, 10, 50, 0),
#         training_window='15m')
#     df = to_pandas(df)

#     assert list(df['id'].values) == [9]
#     assert list(df['value'].values) == [0]


# def test_query_by_indexed_variable(es):
#     df = es.query_by_values(
#         entity_id='log',
#         instance_vals=['taco clock'],
#         variable_id='product_id')
#     df = to_pandas(df)

#     assert list(df['id'].values) == [15, 16]


@pytest.fixture
def pd_df():
    return pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'c']})


@pytest.fixture
def dd_df(pd_df):
    return dd.from_pandas(pd_df, npartitions=2)


@pytest.fixture
def ks_df(pd_df):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_df)


@pytest.fixture(params=['pd_df', 'dd_df', 'ks_df'])
def df(request):
    return request.getfixturevalue(request.param)


def test_check_variables_and_dataframe(df):
    # matches
    logical_types = {'id': ltypes.Integer,
                     'category': ltypes.Categorical}
    es = EntitySet(id='test')
    es.add_dataframe('test_dataframe', df, index='id',
                     logical_types=logical_types)
    assert es.dataframe_dict['test_dataframe'].ww.logical_types['category'] == ltypes.Categorical
    assert es.dataframe_dict['test_dataframe'].ww.semantic_tags['category'] == {'category'}


def test_make_index_any_location(df):
    if ks and isinstance(df, ks.DataFrame):
        pytest.xfail('Woodwork does not support make_index on Koalas DataFrames')
    logical_types = {'id': ltypes.Integer,
                     'category': ltypes.Categorical}

    es = EntitySet(id='test')
    es.add_dataframe(dataframe_id='test_dataframe',
                     index='id1',
                     make_index=True,
                     logical_types=logical_types,
                     dataframe=df)
    if dd and isinstance(df, dd.DataFrame):
        assert es.dataframe_dict['test_dataframe'].columns[-1] == 'id1'
    else:
        assert es.dataframe_dict['test_dataframe'].columns[0] == 'id1'

    assert es.dataframe_dict['test_dataframe'].ww.index == 'id1'


def test_index_any_location(df):
    logical_types = {'id': ltypes.Integer,
                     'category': ltypes.Categorical}

    es = EntitySet(id='test')
    es.add_dataframe(dataframe_id='test_dataframe',
                     index='category',
                     logical_types=logical_types,
                     dataframe=df)
    assert es.dataframe_dict['test_dataframe'].columns[1] == 'category'
    assert es.dataframe_dict['test_dataframe'].ww.index == 'category'


def test_extra_variable_type(df):
    # more variables
    logical_types = {'id': ltypes.Integer,
                     'category': ltypes.Categorical,
                     'category2': ltypes.Categorical}

    error_text = re.escape("logical_types contains columns that are not present in dataframe: ['category2']")
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id='test')
        es.add_dataframe(dataframe_id='test_dataframe',
                         index='id',
                         logical_types=logical_types, dataframe=df)


# def test_add_parent_not_index_varible(es):
#     error_text = "Parent column.*is not the index of dataframe Entity.*"
#     with pytest.raises(AttributeError, match=error_text):
#         es.add_relationship(u'régions', 'language', 'customers', u'région_id')


@pytest.fixture
def pd_df2():
    return pd.DataFrame({'category': [1, 2, 3], 'category2': ['1', '2', '3']})


@pytest.fixture
def dd_df2(pd_df2):
    return dd.from_pandas(pd_df2, npartitions=2)


@pytest.fixture
def ks_df2(pd_df2):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_df2)


@pytest.fixture(params=['pd_df2', 'dd_df2', 'ks_df2'])
def df2(request):
    return request.getfixturevalue(request.param)


def test_none_index(df2):
    es = EntitySet(id='test')

    copy_df = df2.copy()
    copy_df.ww.init()
    error_msg = 'Cannot add Woodwork DataFrame to EntitySet without index'
    with pytest.raises(ValueError, match=error_msg):
        es.add_dataframe(dataframe_id='test_dataframe', dataframe=copy_df)

    warn_text = "Using first column as index. To change this, specify the index parameter"
    with pytest.warns(UserWarning, match=warn_text):
        es.add_dataframe(dataframe_id='test_dataframe',
                         logical_types={'category': 'Categorical'},
                         dataframe=df2)
    assert es['test_dataframe'].ww.index == 'category'
    assert es['test_dataframe'].ww.semantic_tags['category'] == {'index'}
    assert es['test_dataframe'].ww.logical_types['category'] == ltypes.Categorical


@pytest.fixture
def pd_df3():
    return pd.DataFrame({'category': [1, 2, 3]})


@pytest.fixture
def dd_df3(pd_df3):
    return dd.from_pandas(pd_df3, npartitions=2)


@pytest.fixture
def ks_df3(pd_df3):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_df3)


@pytest.fixture(params=['pd_df3', 'dd_df3', 'ks_df3'])
def df3(request):
    return request.getfixturevalue(request.param)


def test_unknown_index(df3):
    error_message = 'Specified index column `id` not found in dataframe. To create a new index column, set make_index to True.'
    es = EntitySet(id='test')
    with pytest.raises(ww.exceptions.ColumnNotPresentError, match=error_message):
        es.add_dataframe(dataframe_id='test_dataframe',
                         index='id',
                         logical_types={'category': 'Categorical'}, dataframe=df3)
# --> test not allowing for koalas index!!!


def test_doesnt_remake_index(df):
    error_text = "When setting make_index to True, the name specified for index cannot match an existing column name"
    with pytest.raises(IndexError, match=error_text):
        es = EntitySet(id='test')
        es.add_dataframe(dataframe_id='test_dataframe',
                         index='id',
                         make_index=True,
                         dataframe=df)


def test_bad_time_index_variable(df3):
    error_text = "Specified time index column `time` not found in dataframe"
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id='test')
        es.add_dataframe(dataframe_id='test_dataframe',
                         dataframe=df3,
                         time_index='time')


@pytest.fixture
def pd_df4():
    df = pd.DataFrame({'id': [0, 1, 2],
                       'category': ['a', 'b', 'a'],
                       'category_int': [1, 2, 3],
                       'ints': ['1', '2', '3'],
                       'floats': ['1', '2', '3.0']})
    df["category_int"] = df["category_int"].astype("category")
    return df


@pytest.fixture
def dd_df4(pd_df4):
    return dd.from_pandas(pd_df4, npartitions=2)


@pytest.fixture
def ks_df4(pd_df4):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_to_ks_clean(pd_df4))


@pytest.fixture(params=['pd_df4', 'dd_df4', 'ks_df4'])
def df4(request):
    return request.getfixturevalue(request.param)


def test_converts_dtype_on_init(df4):
    logical_types = {'id': ltypes.Integer,
                     'ints': ltypes.Integer,
                     'floats': ltypes.Double}
    if not isinstance(df4, pd.DataFrame):
        logical_types['category'] = ltypes.Categorical
        logical_types['category_int'] = ltypes.Categorical
    es = EntitySet(id='test')
    df4.ww.init(index='id', logical_types=logical_types)
    es.add_dataframe(dataframe_id='test_dataframe', dataframe=df4)

    entity_df = es['test_dataframe']
    assert entity_df['ints'].dtype.name == 'int64'
    assert entity_df['floats'].dtype.name == 'float64'

    # this is infer from pandas dtype
    df = es["test_dataframe"]
    assert df.ww.logical_types['category_int'] == ltypes.Categorical


def test_converts_dtype_after_init(df4):
    category_dtype = 'category'
    if ks and isinstance(df4, ks.DataFrame):
        category_dtype = 'string'

    df4["category"] = df4["category"].astype(category_dtype)
    if not isinstance(df4, pd.DataFrame):
        logical_types = {'id': ltypes.Integer,
                         'category': ltypes.Categorical,
                         'category_int': ltypes.Categorical,
                         'ints': ltypes.Integer,
                         'floats': ltypes.Double}
    else:
        logical_types = None
    es = EntitySet(id='test')
    es.add_dataframe(dataframe_id='test_dataframe', index='id',
                     dataframe=df4, logical_types=logical_types)
    df = es['test_dataframe']

    df.ww.set_types(logical_types={'ints': 'Integer'})
    assert df.ww.logical_types['ints'] == ltypes.Integer
    assert df['ints'].dtype == 'int64'

    df.ww.set_types(logical_types={'ints': 'Categorical'})
    assert df.ww.logical_types['ints'] == ltypes.Categorical
    assert df['ints'].dtype == category_dtype

    df.ww.set_types(logical_types={'ints': ltypes.Ordinal(order=[1, 2, 3])})
    assert df.ww.logical_types['ints'] == ltypes.Ordinal(order=[1, 2, 3])
    assert df['ints'].dtype == category_dtype

    df.ww.set_types(logical_types={'ints': 'NaturalLanguage'})
    assert df.ww.logical_types['ints'] == ltypes.NaturalLanguage
    assert df['ints'].dtype == 'string'


def test_warns_no_typing(df4):
    es = EntitySet(id='test')
    if not isinstance(df4, pd.DataFrame):
        msg = 'Performing type inference on Dask or Koalas DataFrames may be computationally intensive. Specify logical types for each column to speed up EntitySet initialization.'
        with pytest.warns(UserWarning, match=msg):
            es.add_dataframe(dataframe_id='test_dataframe', index='id',
                             dataframe=df4)
    else:
        es.add_dataframe(dataframe_id='test_dataframe', index='id',
                         dataframe=df4)

    assert 'test_dataframe' in es.dataframe_dict


@pytest.fixture
def pd_datetime1():
    times = pd.date_range('1/1/2011', periods=3, freq='H')
    time_strs = times.strftime('%Y-%m-%d')
    return pd.DataFrame({'id': [0, 1, 2], 'time': time_strs})


@pytest.fixture
def dd_datetime1(pd_datetime1):
    return dd.from_pandas(pd_datetime1, npartitions=2)


@pytest.fixture
def ks_datetime1(pd_datetime1):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_datetime1)


@pytest.fixture(params=['pd_datetime1', 'dd_datetime1', 'ks_datetime1'])
def datetime1(request):
    return request.getfixturevalue(request.param)


def test_converts_datetime(datetime1):
    # string converts to datetime correctly
    # This test fails without defining vtypes.  Entityset
    # infers time column should be numeric type
    logical_types = {'id': ltypes.Integer,
                     'time': ltypes.Datetime}

    es = EntitySet(id='test')
    es.add_dataframe(
        dataframe_id='test_dataframe',
        index='id',
        time_index="time",
        logical_types=logical_types,
        dataframe=datetime1)
    pd_col = to_pandas(es['test_dataframe']['time'])
    # assert type(es['test_entity']['time']) == variable_types.Datetime
    assert type(pd_col[0]) == pd.Timestamp


@pytest.fixture
def pd_datetime2():
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp('Jan 2, 2011')
    time_strs = [actual.strftime(datetime_format)] * 3
    return pd.DataFrame(
        {'id': [0, 1, 2], 'time_format': time_strs, 'time_no_format': time_strs})


@pytest.fixture
def dd_datetime2(pd_datetime2):
    return dd.from_pandas(pd_datetime2, npartitions=2)


@pytest.fixture
def ks_datetime2(pd_datetime2):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_datetime2)


@pytest.fixture(params=['pd_datetime2', 'dd_datetime2', 'ks_datetime2'])
def datetime2(request):
    return request.getfixturevalue(request.param)


def test_handles_datetime_format(datetime2):
    # check if we load according to the format string
    # pass in an ambigious date
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp('Jan 2, 2011')

    logical_types = {'id': ltypes.Integer,
                     'time_format': (ltypes.Datetime(datetime_format=datetime_format)),
                     'time_no_format': ltypes.Datetime}

    es = EntitySet(id='test')
    es.add_dataframe(
        dataframe_id='test_dataframe',
        index='id',
        logical_types=logical_types,
        dataframe=datetime2)

    col_format = to_pandas(es['test_dataframe']['time_format'])
    col_no_format = to_pandas(es['test_dataframe']['time_no_format'])
    # without formatting pandas gets it wrong
    assert (col_no_format != actual).all()

    # with formatting we correctly get jan2
    assert (col_format == actual).all()


# Inferring variable types and verifying typing not supported in Dask, Koalas
def test_handles_datetime_mismatch():
    # can't convert arbitrary strings
    df = pd.DataFrame({'id': [0, 1, 2], 'time': ['a', 'b', 'tomorrow']})
    logical_types = {'id': ltypes.Integer,
                     'time': ltypes.Datetime}

    error_text = "Time index column must contain datetime or numeric values"
    with pytest.raises(TypeError, match=error_text):
        es = EntitySet(id='test')
        es.add_dataframe('test_dataframe', df, 'id',
                         time_index='time', logical_types=logical_types)


def test_dataframe_init(es):
    # Note: to convert the time column directly either the variable type
    # or convert_date_columns must be specifie
    df = pd.DataFrame({'id': ['0', '1', '2'],
                       'time': [datetime(2011, 4, 9, 10, 31, 3 * i)
                                for i in range(3)],
                       'category': ['a', 'b', 'a'],
                       'number': [4, 5, 6]})
    if any(isinstance(dataframe, dd.DataFrame) for dataframe in es.dataframes):
        df = dd.from_pandas(df, npartitions=2)
    if ks and any(isinstance(dataframe, ks.DataFrame) for dataframe in es.dataframes):
        df = ks.from_pandas(df)

    logical_types = {'time': ltypes.Datetime}
    if not isinstance(df, pd.DataFrame):
        extra_logical_types = {
            'id': ltypes.Categorical,
            'category': ltypes.Categorical,
            'number': ltypes.Integer
        }
        logical_types.update(extra_logical_types)
    es.add_dataframe('test_dataframe', df.copy(), index='id',
                     time_index='time', logical_types=logical_types)
    if isinstance(df, dd.DataFrame):
        df_shape = (df.shape[0].compute(), df.shape[1])
    else:
        df_shape = df.shape
    if isinstance(es['test_dataframe'], dd.DataFrame):
        es_df_shape = (es['test_dataframe'].shape[0].compute(), es['test_dataframe'].shape[1])
    else:
        es_df_shape = es['test_dataframe'].shape
    assert es_df_shape == df_shape
    assert es['test_dataframe'].ww.index == 'id'
    assert es['test_dataframe'].ww.time_index == 'time'
    assert set([v for v in es['test_dataframe'].ww.columns]) == set(df.columns)

    assert es['test_dataframe']['time'].dtype == df['time'].dtype
    if ks and isinstance(es['test_dataframe'], ks.DataFrame):
        assert set(es['test_dataframe']['id'].to_list()) == set(df['id'].to_list())
    else:
        assert set(es['test_dataframe']['id']) == set(df['id'])


@pytest.fixture
def pd_bad_df():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 3: ['a', 'b', 'c']})


@pytest.fixture
def dd_bad_df(pd_bad_df):
    return dd.from_pandas(pd_bad_df, npartitions=2)


@pytest.fixture(params=['pd_bad_df', 'dd_bad_df'])
def bad_df(request):
    return request.getfixturevalue(request.param)


# Skip for Koalas, automatically converts non-str column names to str
def test_nonstr_column_names(bad_df):
    if dd and isinstance(bad_df, dd.DataFrame):
        pytest.xfail('Dask DataFrames cannot handle integer column names')

    es = ft.EntitySet(id='Failure')
    error_text = r"All column names must be strings \(Columns \[3\] are not strings\)"
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_id='str_cols',
                         dataframe=bad_df,
                         index='a')

    bad_df.ww.init()
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_id='str_cols',
                         dataframe=bad_df)


def test_sort_time_id():
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s")[::-1]})

    es = EntitySet("test", dataframes={"t": (transactions_df.copy(), "id", "transaction_time")})
    assert es['t'] is not transactions_df
    times = list(es["t"].transaction_time)
    assert times == sorted(list(transactions_df.transaction_time))


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
    es.add_dataframe('t',
                     transactions_df.copy(),
                     index='id',
                     time_index="transaction_time",
                     already_sorted=True)

    assert es['t'] is not transactions_df
    times = list(es["t"].transaction_time)
    assert times == list(transactions_df.transaction_time)


# # TODO: equality check fails, dask series have no .equals method; error computing lti if categorical index
# # TODO: dask deepcopy
# def test_concat_entitysets(es):
#     df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
#     if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
#         pytest.xfail("Dask has no .equals method and issue with categoricals "
#                      "and add_last_time_indexes")

#     if ks and any(isinstance(entity.df, ks.DataFrame) for entity in es.entities):
#         pytest.xfail("Koalas deepcopy fails")

#     vtypes = {'id': variable_types.Categorical,
#               'category': variable_types.Categorical}
#     es.entity_from_dataframe(entity_id='test_entity',
#                              index='id1',
#                              make_index=True,
#                              variable_types=vtypes,
#                              dataframe=df)
#     es.add_last_time_indexes()

#     assert es.__eq__(es)
#     es_1 = copy.deepcopy(es)
#     es_2 = copy.deepcopy(es)

#     # map of what rows to take from es_1 and es_2 for each entity
#     emap = {
#         'log': [list(range(10)) + [14, 15, 16], list(range(10, 14)) + [15, 16]],
#         'sessions': [[0, 1, 2, 5], [1, 3, 4, 5]],
#         'customers': [[0, 2], [1, 2]],
#         'test_entity': [[0, 1], [0, 2]],
#     }

#     assert es.__eq__(es_1, deep=True)
#     assert es.__eq__(es_2, deep=True)

#     for i, _es in enumerate([es_1, es_2]):
#         for entity, rows in emap.items():
#             df = _es[entity].df
#             _es.update_dataframe(entity_id=entity, df=df.loc[rows[i]])

#     assert 10 not in es_1['log'].last_time_index.index
#     assert 10 in es_2['log'].last_time_index.index
#     assert 9 in es_1['log'].last_time_index.index
#     assert 9 not in es_2['log'].last_time_index.index
#     assert not es.__eq__(es_1, deep=True)
#     assert not es.__eq__(es_2, deep=True)

#     # make sure internal indexes work before concat
#     regions = es_1.query_by_values('customers', ['United States'], variable_id=u'région_id')
#     assert regions.index.isin(es_1['customers'].df.index).all()

#     assert es_1.__eq__(es_2)
#     assert not es_1.__eq__(es_2, deep=True)

#     old_es_1 = copy.deepcopy(es_1)
#     old_es_2 = copy.deepcopy(es_2)
#     es_3 = es_1.concat(es_2)

#     assert old_es_1.__eq__(es_1, deep=True)
#     assert old_es_2.__eq__(es_2, deep=True)

#     assert es_3.__eq__(es)
#     for entity in es.entities:
#         df = es[entity.id].df.sort_index()
#         df_3 = es_3[entity.id].df.sort_index()
#         for column in df:
#             for x, y in zip(df[column], df_3[column]):
#                 assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
#         orig_lti = es[entity.id].last_time_index.sort_index()
#         new_lti = es_3[entity.id].last_time_index.sort_index()
#         for x, y in zip(orig_lti, new_lti):
#             assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

#     es_1['stores'].last_time_index = None
#     es_1['test_entity'].last_time_index = None
#     es_2['test_entity'].last_time_index = None
#     es_4 = es_1.concat(es_2)
#     assert not es_4.__eq__(es, deep=True)
#     for entity in es.entities:
#         df = es[entity.id].df.sort_index()
#         df_4 = es_4[entity.id].df.sort_index()
#         for column in df:
#             for x, y in zip(df[column], df_4[column]):
#                 assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

#         if entity.id != 'test_entity':
#             orig_lti = es[entity.id].last_time_index.sort_index()
#             new_lti = es_4[entity.id].last_time_index.sort_index()
#             for x, y in zip(orig_lti, new_lti):
#                 assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
#         else:
#             assert es_4[entity.id].last_time_index is None


@pytest.fixture
def pd_transactions_df():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                         "card_id": [1, 2, 1, 3, 4, 5],
                         "transaction_time": [10, 12, 13, 20, 21, 20],
                         "fraud": [True, False, False, False, True, True]})


@pytest.fixture
def dd_transactions_df(pd_transactions_df):
    return dd.from_pandas(pd_transactions_df, npartitions=3)


@pytest.fixture
def ks_transactions_df(pd_transactions_df):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_transactions_df)


@pytest.fixture(params=['pd_transactions_df', 'dd_transactions_df', 'ks_transactions_df'])
def transactions_df(request):
    return request.getfixturevalue(request.param)


def test_set_time_type_on_init(transactions_df):
    # create cards entity
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    if isinstance(transactions_df, dd.DataFrame):
        cards_df = dd.from_pandas(cards_df, npartitions=3)
    if ks and isinstance(transactions_df, ks.DataFrame):
        cards_df = ks.from_pandas(cards_df)
    if not isinstance(transactions_df, pd.DataFrame):
        cards_logical_types = {'id': ltypes.Categorical}
        transactions_logical_types = {
            'id': ltypes.Integer,
            'card_id': ltypes.Categorical,
            'transaction_time': ltypes.Integer,
            'fraud': ltypes.Boolean
        }
    else:
        cards_logical_types = None
        transactions_logical_types = None

    entities = {
        "cards": (cards_df, "id", None, cards_logical_types),
        "transactions": (transactions_df, "id", "transaction_time", transactions_logical_types)
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    es = EntitySet("fraud", entities, relationships)
    # assert time_type is set
    assert es.time_type == 'numeric'


def test_sets_time_when_adding_entity(transactions_df):
    accounts_df = pd.DataFrame({"id": [3, 4, 5],
                                "signup_date": [datetime(2002, 5, 1),
                                                datetime(2006, 3, 20),
                                                datetime(2011, 11, 11)]})
    accounts_df_string = pd.DataFrame({"id": [3, 4, 5],
                                       "signup_date": ["element",
                                                       "exporting",
                                                       "editable"]})
    if isinstance(transactions_df, dd.DataFrame):
        accounts_df = dd.from_pandas(accounts_df, npartitions=2)
    if ks and isinstance(transactions_df, ks.DataFrame):
        accounts_df = ks.from_pandas(accounts_df)
    if not isinstance(transactions_df, pd.DataFrame):
        accounts_logical_types = {'id': ltypes.Categorical, 'signup_date': ltypes.Datetime}
        transactions_logical_types = {
            'id': ltypes.Integer,
            'card_id': ltypes.Categorical,
            'transaction_time': ltypes.Integer,
            'fraud': ltypes.Boolean
        }
    else:
        accounts_logical_types = None
        transactions_logical_types = None

    # create empty entityset
    es = EntitySet("fraud")
    # assert it's not set
    assert getattr(es, "time_type", None) is None
    # add entity
    es.add_dataframe("transactions",
                     transactions_df,
                     index="id",
                     time_index="transaction_time",
                     logical_types=transactions_logical_types)
    # assert time_type is set
    assert es.time_type == 'numeric'
    # add another entity
    es.normalize_dataframe("transactions",
                           "cards",
                           "card_id",
                           make_time_index=True)
    # assert time_type unchanged
    assert es.time_type == 'numeric'
    # add wrong time type entity
    error_text = "accounts time index is Datetime type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.add_dataframe("accounts",
                         accounts_df,
                         index="id",
                         time_index="signup_date",
                         logical_types=accounts_logical_types)
    # add non time type as time index, only valid for pandas
    if isinstance(transactions_df, pd.DataFrame):
        error_text = "Time index column must contain datetime or numeric values"
        with pytest.raises(TypeError, match=error_text):
            es.add_dataframe("accounts",
                             accounts_df_string,
                             index="id",
                             time_index="signup_date")


def test_checks_time_type_setting_time_index(es):
    error_text = 'Time index column must be a Datetime or numeric column.'
    with pytest.raises(TypeError, match=error_text):
        es['log'].ww.set_time_index('purchased')
        es._check_uniform_time_index(es['log'])


def test_checks_time_type_setting_secondary_time_index(es):
    # entityset is timestamp time type
    assert es.time_type == ltypes.Datetime
    # add secondary index that is timestamp type
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'cancel_date': ['cancel_date', 'cancel_reason']}
    es.set_secondary_time_index(es["customers"], new_2nd_ti)
    assert es.time_type == ltypes.Datetime
    # add secondary index that is numeric type
    new_2nd_ti = {'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index(es["customers"], new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {'favorite_quote': ['favorite_quote', 'loves_ice_cream']}

    error_text = 'customers time index not recognized as numeric or datetime'
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index(es["customers"], new_2nd_ti)
    # add mismatched pair of secondary time indexes
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index(es["customers"], new_2nd_ti)

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
    assert card_es.time_type == 'numeric'
    # add secondary index that is numeric time type
    new_2nd_ti = {'fraud_decision_time': ['fraud_decision_time', 'fraud']}
    card_es.set_secondary_time_index(card_es['transactions'], new_2nd_ti)
    assert card_es.time_type == 'numeric'
    # add secondary index that is timestamp type
    new_2nd_ti = {'transaction_date': ['transaction_date', 'fraud']}

    error_text = "transactions time index is Datetime type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index(card_es['transactions'], new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {'transaction_city': ['transaction_city', 'fraud']}

    error_text = 'transactions time index not recognized as numeric or datetime'
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index(card_es['transactions'], new_2nd_ti)
    # add mixed secondary time indexes
    new_2nd_ti = {'transaction_city': ['transaction_city', 'fraud'],
                  'fraud_decision_time': ['fraud_decision_time', 'fraud']}
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index(card_es['transactions'], new_2nd_ti)

    # add bool secondary time index
    error_text = 'transactions time index not recognized as numeric or datetime'
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index(card_es['transactions'], {'fraud': ['fraud']})


def test_normalize_dataframe(es):
    error_text = "'additional_columns' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe('sessions', 'device_types', 'device_type',
                               additional_columns='log')

    error_text = "'copy_columns' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe('sessions', 'device_types', 'device_type',
                               copy_columns='log')

    es.normalize_dataframe('sessions', 'device_types', 'device_type',
                           additional_columns=['device_name'],
                           make_time_index=False)

    assert len(es.get_forward_relationships('sessions')) == 2
    assert es.get_forward_relationships(
        'sessions')[1].parent_dataframe.ww.name == 'device_types'
    assert 'device_name' in es['device_types'].columns
    assert 'device_name' not in es['sessions'].columns
    assert 'device_type' in es['device_types'].columns


def test_normalize_dataframe_new_time_index_in_base_entity_error_check(es):
    error_text = "'make_time_index' must be a variable in the base entity"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(base_dataframe_id='customers',
                               new_dataframe_id='cancellations',
                               index='cancel_reason',
                               make_time_index="non-existent")


def test_normalize_entity_new_time_index_in_variable_list_error_check(es):
    error_text = "'make_time_index' must be specified in 'additional_columns' or 'copy_columns'"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(base_dataframe_id='customers',
                               new_dataframe_id='cancellations',
                               index='cancel_reason',
                               make_time_index='cancel_date')


def test_normalize_dataframe_new_time_index_copy_success_check(es):
    es.normalize_dataframe(base_dataframe_id='customers',
                           new_dataframe_id='cancellations',
                           index='cancel_reason',
                           make_time_index='cancel_date',
                           additional_columns=[],
                           copy_columns=['cancel_date'])


def test_normalize_dataframe_new_time_index_additional_success_check(es):
    es.normalize_dataframe(base_dataframe_id='customers',
                           new_dataframe_id='cancellations',
                           index='cancel_reason',
                           make_time_index='cancel_date',
                           additional_columns=['cancel_date'],
                           copy_columns=[])


@pytest.fixture
def pd_normalize_es():
    df = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "A": [5, 4, 2, 3],
        'time': [datetime(2020, 6, 3), (datetime(2020, 3, 12)), datetime(2020, 5, 1), datetime(2020, 4, 22)]
    })
    es = ft.EntitySet("es")
    return es.add_dataframe(
        dataframe_id="data",
        dataframe=df,
        index="id")


@pytest.fixture
def dd_normalize_es(pd_normalize_es):
    es = ft.EntitySet(id=pd_normalize_es.id)
    dd_df = dd.from_pandas(pd_normalize_es['data'], npartitions=2)
    dd_df.ww.init(schema=pd_normalize_es['data'].ww.schema)

    es.add_dataframe(dataframe_id=dd_df.ww.name,
                     dataframe=dd_df)
    return es


@pytest.fixture
def ks_normalize_es(pd_normalize_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    es = ft.EntitySet(id=pd_normalize_es.id)
    ks_df = ks.from_pandas(pd_normalize_es['data'])
    ks_df.ww.init(schema=pd_normalize_es['data'].ww.schema)
    es.add_dataframe(dataframe_id=ks_df.ww.name,
                     dataframe=ks_df)
    return es


@pytest.fixture(params=['pd_normalize_es', 'dd_normalize_es', 'ks_normalize_es'])
def normalize_es(request):
    return request.getfixturevalue(request.param)


def test_normalize_time_index_from_none(normalize_es):
    assert normalize_es['data'].ww.time_index is None

    normalize_es.normalize_dataframe(base_dataframe_id='data',
                                     new_dataframe_id='normalized',
                                     index='A',
                                     make_time_index='time',
                                     copy_columns=['time'])
    assert normalize_es['normalized'].ww.time_index == 'time'
    df = normalize_es['normalized']

    # only pandas sorts by time index
    if isinstance(df, pd.DataFrame):
        assert df['time'].is_monotonic_increasing


def test_raise_error_if_dupicate_additional_variables_passed(es):
    error_text = "'additional_columns' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe('sessions', 'device_types', 'device_type',
                               additional_columns=['device_name', 'device_name'])


def test_raise_error_if_dupicate_copy_variables_passed(es):
    error_text = "'copy_columns' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe('sessions', 'device_types', 'device_type',
                               copy_columns=['device_name', 'device_name'])


def test_normalize_dataframe_copies_variable_types(es):
    es['log'].ww.set_types(logical_types={'value': ltypes.Ordinal(order=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 15.0, 20.0])})

    assert isinstance(es['log'].ww.logical_types['value'], ltypes.Ordinal)
    assert isinstance(es['log'].ww.logical_types['priority_level'], ltypes.Ordinal)
    es.normalize_dataframe('log', 'values_2', 'value_2',
                           additional_columns=['priority_level'],
                           copy_columns=['value'],
                           make_time_index=False)

    assert len(es.get_forward_relationships('log')) == 3
    assert es.get_forward_relationships(
        'log')[2].parent_dataframe.ww.name == 'values_2'
    assert 'priority_level' in es['values_2'].columns
    assert 'value' in es['values_2'].columns
    assert 'priority_level' not in es['log'].columns
    assert 'value' in es['log'].columns
    assert 'value_2' in es['values_2'].columns
    assert isinstance(es['values_2'].ww.logical_types['priority_level'], ltypes.Ordinal)
    assert isinstance(es['values_2'].ww.logical_types['value'], ltypes.Ordinal)


# sorting not supported in Dask, Koalas
def test_make_time_index_keeps_original_sorting():
    trips = {
        'trip_id': [999 - i for i in range(1000)],
        'flight_time': [datetime(1997, 4, 1) for i in range(1000)],
        'flight_id': [1 for i in range(350)] + [2 for i in range(650)]
    }
    order = [i for i in range(1000)]
    df = pd.DataFrame.from_dict(trips)
    es = EntitySet('flights')
    es.add_dataframe("trips",
                     dataframe=df,
                     index="trip_id",
                     time_index='flight_time')
    assert (es['trips']['trip_id'] == order).all()
    es.normalize_dataframe(base_dataframe_id="trips",
                           new_dataframe_id="flights",
                           index="flight_id",
                           make_time_index=True)
    assert (es['trips']['trip_id'] == order).all()


def test_normalize_dataframe_new_time_index(es):
    new_time_index = 'value_time'
    es.normalize_dataframe('log', 'values', 'value',
                           make_time_index=True,
                           new_dataframe_time_index=new_time_index)

    assert es['values'].ww.time_index == new_time_index
    assert new_time_index in es['values'].columns
    assert len(es['values'].columns) == 2
    df = to_pandas(es['values'], sort_index=True)
    assert df[new_time_index].is_monotonic_increasing


def test_normalize_dataframe_same_index(es):
    transactions_df = pd.DataFrame({"id": [1, 2, 3],
                                    "transaction_time": pd.date_range(start="10:00", periods=3, freq="10s"),
                                    "first_df_time": [1, 2, 3]})
    es = ft.EntitySet("example")
    es.add_dataframe(dataframe_id="df",
                     index="id",
                     time_index="transaction_time",
                     dataframe=transactions_df)

    error_text = "'index' must be different from the index column of the base entity"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(base_dataframe_id="df",
                               new_dataframe_id="new_dataframe",
                               index="id",
                               make_time_index=True)


def test_secondary_time_index(es):
    es.normalize_dataframe('log', 'values', 'value',
                           make_time_index=True,
                           make_secondary_time_index={
                               'datetime': ['comments']},
                           new_dataframe_time_index="value_time",
                           new_dataframe_secondary_time_index='second_ti')

    assert (es['values'].ww.logical_types['second_ti'] == ltypes.Datetime)
    assert (es['values'].ww.semantic_tags['second_ti'] == set())
    assert (es['values'].ww.metadata['secondary_time_index'] == {
            'second_ti': ['comments', 'second_ti']})


# --> wait till after implement last time index
# def test_sizeof(es):
#     total_size = 0
#     for entity in es.entities:
#         total_size += entity.df.__sizeof__()
#         total_size += entity.last_time_index.__sizeof__()

#     assert es.__sizeof__() == total_size


def test_construct_without_id():
    assert ft.EntitySet().id is None


def test_repr_without_id():
    match = 'Entityset: None\n  Entities:\n  Relationships:\n    No relationships'
    assert repr(ft.EntitySet()) == match


def test_getitem_without_id():
    error_text = 'DataFrame test does not exist in entity set'
    with pytest.raises(KeyError, match=error_text):
        ft.EntitySet()['test']


# --> wait till serialization implemented
# def test_metadata_without_id():
#     es = ft.EntitySet()
#     assert es.metadata.id is None


@pytest.fixture
def pd_datetime3():
    return pd.DataFrame({'id': [0, 1, 2],
                         'ints': ['1', '2', '1']})


@pytest.fixture
def dd_datetime3(pd_datetime3):
    return dd.from_pandas(pd_datetime3, npartitions=2)


@pytest.fixture
def ks_datetime3(pd_datetime3):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_datetime3)


@pytest.fixture(params=['pd_datetime3', 'dd_datetime3', 'ks_datetime3'])
def datetime3(request):
    return request.getfixturevalue(request.param)


def test_datetime64_conversion(datetime3):
    df = datetime3
    df["time"] = pd.Timestamp.now()
    if ks and isinstance(df, ks.DataFrame):
        df['time'] = df['time'].astype(np.datetime64)
    else:
        df["time"] = df["time"].astype("datetime64[ns, UTC]")

    if not isinstance(df, pd.DataFrame):
        logical_types = {
            'id': ltypes.Integer,
            'ints': ltypes.Integer,
            'time': ltypes.Datetime
        }
    else:
        logical_types = None
    es = EntitySet(id='test')
    es.add_dataframe(dataframe_id='test_dataframe',
                     index='id',
                     dataframe=df,
                     logical_types=logical_types)
    # --> is this testing setting time index or setting ltype??? and why are there no checks afterwards
    es['test_dataframe'].ww.set_time_index('time')
    assert es['test_dataframe'].ww.time_index == 'time'


@pytest.fixture
def pd_index_df():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                         "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s"),
                         "first_entity_time": [1, 2, 3, 5, 6, 6]})


@pytest.fixture
def dd_index_df(pd_index_df):
    return dd.from_pandas(pd_index_df, npartitions=3)


@pytest.fixture
def ks_index_df(pd_index_df):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    return ks.from_pandas(pd_index_df)


@pytest.fixture(params=['pd_index_df', 'dd_index_df', 'ks_index_df'])
def index_df(request):
    return request.getfixturevalue(request.param)


def test_same_index_values(index_df):
    if not isinstance(index_df, pd.DataFrame):
        logical_types = {
            'id': ltypes.Integer,
            'transaction_time': ltypes.Datetime,
            'first_entity_time': ltypes.Integer
        }
    else:
        logical_types = None

    es = ft.EntitySet("example")

    error_text = '"id" is already set as the index. An index cannot also be the time index.'
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_id="entity",
                         index="id",
                         time_index="id",
                         dataframe=index_df,
                         logical_types=logical_types)

    es.add_dataframe(dataframe_id="entity",
                     index="id",
                     time_index="transaction_time",
                     dataframe=index_df,
                     logical_types=logical_types)

    error_text = "time_index and index cannot be the same value, first_entity_time"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(base_dataframe_id="entity",
                               new_dataframe_id="new_entity",
                               index="first_entity_time",
                               make_time_index=True)


def test_use_time_index(index_df):
    if not isinstance(index_df, pd.DataFrame):
        bad_vtypes = {
            'id': ltypes.Integer,
            'transaction_time': ltypes.Datetime,
            'first_entity_time': ltypes.Integer
        }
        bad_semantic_tags = {'transaction_time': 'time_index'}
        logical_types = {
            'id': ltypes.Integer,
            'transaction_time': ltypes.Datetime,
            'first_entity_time': ltypes.Integer
        }
    else:
        bad_vtypes = {"transaction_time": ltypes.Datetime}
        bad_semantic_tags = {'transaction_time': 'time_index'}
        logical_types = None

    es = ft.EntitySet()

    error_text = re.escape("Cannot add 'time_index' tag directly for column transaction_time. To set a column as the time index, use DataFrame.ww.set_time_index() instead.")
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_id="entity",
                         index="id",
                         logical_types=bad_vtypes,
                         semantic_tags=bad_semantic_tags,
                         dataframe=index_df)

    es.add_dataframe(dataframe_id="entity",
                     index="id",
                     time_index="transaction_time",
                     logical_types=logical_types,
                     dataframe=index_df)


def test_normalize_with_datetime_time_index(es):
    es.normalize_dataframe(base_dataframe_id="customers",
                           new_dataframe_id="cancel_reason",
                           index="cancel_reason",
                           make_time_index=False,
                           copy_columns=['signup_date', 'upgrade_date'])

    assert es['cancel_reason'].ww.logical_types['signup_date'] == ltypes.Datetime
    assert es['cancel_reason'].ww.logical_types['upgrade_date'] == ltypes.Datetime


def test_normalize_with_numeric_time_index(int_es):
    int_es.normalize_dataframe(base_dataframe_id="customers",
                               new_dataframe_id="cancel_reason",
                               index="cancel_reason",
                               make_time_index=False,
                               copy_columns=['signup_date', 'upgrade_date'])

    assert int_es['cancel_reason'].ww.semantic_tags['signup_date'] == {'numeric'}


def test_normalize_with_invalid_time_index(es):
    error_text = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe(base_dataframe_id="customers",
                               new_dataframe_id="cancel_reason",
                               index="cancel_reason",
                               copy_columns=['upgrade_date', 'favorite_quote'],
                               make_time_index='favorite_quote')


def test_entityset_init():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "upgrade_date": [51, 23, 45, 12, 22, 53],
                                    "fraud": [True, False, False, False, True, True]})
    logical_types = {
        'fraud': 'boolean',
        'card_id': 'categorical'
    }
    dataframes = {
        "cards": (cards_df, "id", None, {'id': 'Categorical'}),
        "transactions": (transactions_df, 'id', 'transaction_time',
                         logical_types, None, False)
    }
    relationships = [('cards', 'id', 'transactions', 'card_id')]
    es = ft.EntitySet(id="fraud_data",
                      dataframes=dataframes,
                      relationships=relationships)
    assert es['transactions'].ww.index == 'id'
    assert es['transactions'].ww.time_index == 'transaction_time'
    es_copy = ft.EntitySet(id="fraud_data")
    es_copy.add_dataframe(dataframe_id='cards',
                          dataframe=cards_df,
                          index='id')
    es_copy.add_dataframe(dataframe_id='transactions',
                          dataframe=transactions_df,
                          index='id',
                          logical_types=logical_types,
                          make_index=False,
                          time_index='transaction_time')
    es_copy.add_relationship('cards', 'id', 'transactions', 'card_id')

    # --> any time we do this equality check it will be deep bc Woodwork doesn't differentiate
    assert es['cards'].ww == es_copy['cards'].ww
    assert es['transactions'].ww == es_copy['transactions'].ww


def test_add_interesting_values_specified_vals(es):
    product_vals = ['coke zero', 'taco clock']
    country_vals = ['AL', 'US']
    interesting_values = {
        'product_id': product_vals,
        'countrycode': country_vals,
    }
    es.add_interesting_values(dataframe_id='log', values=interesting_values)

    assert es['log'].ww['product_id'].ww.metadata['interesting_values'] == product_vals
    assert es['log'].ww['countrycode'].ww.metadata['interesting_values'] == country_vals


def test_add_interesting_values_vals_specified_without_entity_id(es):
    interesting_values = {
        'countrycode': ['AL', 'US'],
    }
    error_msg = "entity_id must be specified if values are provided"
    with pytest.raises(ValueError, match=error_msg):
        es.add_interesting_values(values=interesting_values)


def test_add_interesting_values_single_entity(es):
    es.add_interesting_values(dataframe_id='log')
    breakpoint()


def test_add_interesting_values_multiple_entities(es):
    es.add_interesting_values()
    breakpoint()


# --> need to update load_retail
# def test_add_interesting_values_verbose_output(caplog):
#     es = ft.demo.load_retail(nrows=200)
#     es['order_products'].convert_variable_type('quantity', ft.variable_types.Discrete)
#     logger = logging.getLogger('featuretools')
#     logger.propagate = True
#     logger_es = logging.getLogger('featuretools.entityset')
#     logger_es.propagate = True
#     es.add_interesting_values(verbose=True, max_values=10)
#     logger.propagate = False
#     logger_es.propagate = False
#     assert 'Variable country: Marking United Kingdom as an interesting value' in caplog.text
#     assert 'Variable quantity: Marking 6 as an interesting value' in caplog.text


def test_entityset_equality(es):
    first_es = EntitySet()
    second_es = EntitySet()
    assert first_es == second_es

    first_es.add_dataframe(dataframe_id='customers',
                           dataframe=es['customers'],
                           index='id',
                           time_index='signup_date',
                           logical_types=es['customers'].ww.logical_types,
                           semantic_tags=es['customers'].ww.semantic_tags)
    assert first_es != second_es

    second_es.add_dataframe(dataframe_id='sessions',
                            dataframe=es['sessions'],
                            index='id',
                            logical_types=es['sessions'].ww.logical_types,
                            semantic_tags=es['sessions'].ww.semantic_tags)
    assert first_es != second_es

    first_es.add_dataframe(dataframe_id='sessions',
                           dataframe=es['sessions'],
                           index='id',
                           logical_types=es['sessions'].ww.logical_types,
                           semantic_tags=es['sessions'].ww.semantic_tags)
    second_es.add_dataframe(dataframe_id='customers',
                            dataframe=es['customers'],
                            index='id',
                            time_index='signup_date',
                            logical_types=es['customers'].ww.logical_types,
                            semantic_tags=es['customers'].ww.semantic_tags)
    assert first_es == second_es

    first_es.add_relationship('customers', 'id', 'sessions', 'customer_id')
    assert first_es != second_es

    second_es.add_relationship('customers', 'id', 'sessions', 'customer_id')
    assert first_es == second_es
