import copy
from datetime import datetime

import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools import variable_types
from featuretools.entityset import (
    EntitySet,
    Relationship,
    deserialize,
    serialize
)
from featuretools.entityset.serialize import SCHEMA_VERSION


def test_normalize_time_index_as_additional_variable(es):
    error_text = "Not moving signup_date as it is the base time index variable."
    with pytest.raises(ValueError, match=error_text):
        assert "signup_date" in es["customers"].df.columns
        es.normalize_entity(base_entity_id='customers',
                            new_entity_id='cancellations',
                            index='cancel_reason',
                            make_time_index='signup_date',
                            additional_variables=['signup_date'],
                            copy_variables=[])


def test_operations_invalidate_metadata(es):
    new_es = ft.EntitySet(id="test")
    # test metadata gets created on access
    assert new_es._data_description is None
    assert new_es.metadata is not None  # generated after access
    assert new_es._data_description is not None
    if isinstance(es['customers'].df, dd.DataFrame):
        customers_vtypes = es["customers"].variable_types
        customers_vtypes['signup_date'] = variable_types.Datetime
    else:
        customers_vtypes = None
    new_es.entity_from_dataframe("customers",
                                 es["customers"].df,
                                 index=es["customers"].index,
                                 variable_types=customers_vtypes)
    if isinstance(es['sessions'].df, dd.DataFrame):
        sessions_vtypes = es["sessions"].variable_types
    else:
        sessions_vtypes = None
    new_es.entity_from_dataframe("sessions",
                                 es["sessions"].df,
                                 index=es["sessions"].index,
                                 variable_types=sessions_vtypes)
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

    # automatically adding interesting values not supported in Dask
    if any(isinstance(entity.df, pd.DataFrame) for entity in new_es.entities):
        new_es.add_interesting_values()
        assert new_es._data_description is None
        assert new_es.metadata is not None
        assert new_es._data_description is not None


def test_reset_metadata(es):
    assert es.metadata is not None
    assert es._data_description is not None
    es.reset_data_description()
    assert es._data_description is None


def test_cannot_re_add_relationships_that_already_exists(es):
    before_len = len(es.relationships)
    es.add_relationship(es.relationships[0])
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        parent_e = es[r.parent_entity.id]
        child_e = es[r.child_entity.id]
        assert type(r.parent_variable) == variable_types.Index
        assert type(r.child_variable) == variable_types.Id
        assert parent_e.df[r.parent_variable.id].dtype == child_e.df[r.child_variable.id].dtype


def test_add_relationship_errors_on_dtype_mismatch(es):
    log_2_df = es['log'].df.copy()
    log_variable_types = {
        'id': variable_types.Categorical,
        'session_id': variable_types.Id,
        'product_id': variable_types.Id,
        'datetime': variable_types.Datetime,
        'value': variable_types.Numeric,
        'value_2': variable_types.Numeric,
        'latlong': variable_types.LatLong,
        'latlong2': variable_types.LatLong,
        'zipcode': variable_types.ZIPCode,
        'countrycode': variable_types.CountryCode,
        'subregioncode': variable_types.SubRegionCode,
        'value_many_nans': variable_types.Numeric,
        'priority_level': variable_types.Ordinal,
        'purchased': variable_types.Boolean,
        'comments': variable_types.Text
    }
    assert set(log_variable_types) == set(log_2_df.columns)
    es.entity_from_dataframe(entity_id='log2',
                             dataframe=log_2_df,
                             index='id',
                             variable_types=log_variable_types,
                             time_index='datetime')

    error_text = u'Unable to add relationship because id in customers is Pandas dtype category and session_id in log2 is Pandas dtype int64.'
    with pytest.raises(ValueError, match=error_text):
        mismatch = Relationship(es[u'customers']['id'], es['log2']['session_id'])
        es.add_relationship(mismatch)


def test_add_relationship_errors_child_v_index(es):
    log_df = es['log'].df.copy()
    log_vtypes = es['log'].variable_types
    es.entity_from_dataframe(entity_id='log2',
                             dataframe=log_df,
                             index='id',
                             variable_types=log_vtypes,
                             time_index='datetime')

    bad_relationship = ft.Relationship(es['log']['id'], es['log2']['id'])
    to_match = "Unable to add relationship because child variable 'id' in 'log2' is also its index"
    with pytest.raises(ValueError, match=to_match):
        es.add_relationship(bad_relationship)


def test_add_relationship_empty_child_convert_dtype(es):
    relationship = ft.Relationship(es["sessions"]["id"], es["log"]["session_id"])
    es['log'].df = pd.DataFrame(columns=es['log'].df.columns)
    assert len(es['log'].df) == 0
    assert es['log'].df['session_id'].dtype == 'object'

    es.relationships.remove(relationship)
    assert(relationship not in es.relationships)

    es.add_relationship(relationship)
    assert es['log'].df['session_id'].dtype == 'int64'


def test_query_by_id(es):
    df = es['log'].query_by_values(instance_vals=[0])
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    assert df['id'].values[0] == 0


def test_query_by_id_with_time(es):
    df = es['log'].query_by_values(
        instance_vals=[0, 1, 2, 3, 4],
        time_last=datetime(2011, 4, 9, 10, 30, 2 * 6))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    assert list(df['id'].values) == [0, 1, 2]


def test_query_by_variable_with_time(es):
    df = es['log'].query_by_values(
        instance_vals=[0, 1, 2], variable_id='session_id',
        time_last=datetime(2011, 4, 9, 10, 50, 0))
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    true_values = [
        i * 5 for i in range(5)] + [i * 1 for i in range(4)] + [0]

    assert list(df['id'].values) == list(range(10))
    assert list(df['value'].values) == true_values


def test_query_by_variable_with_training_window(es):
    df = es['log'].query_by_values(
        instance_vals=[0, 1, 2], variable_id='session_id',
        time_last=datetime(2011, 4, 9, 10, 50, 0),
        training_window='15m')
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    assert list(df['id'].values) == [9]
    assert list(df['value'].values) == [0]


def test_query_by_indexed_variable(es):
    df = es['log'].query_by_values(
        instance_vals=['taco clock'],
        variable_id='product_id')
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    assert list(df['id'].values) == [15, 16]


@pytest.fixture
def pd_df():
    return pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'c']})


@pytest.fixture
def dd_df(pd_df):
    return dd.from_pandas(pd_df, npartitions=2)


@pytest.fixture(params=['pd_df', 'dd_df'])
def df(request):
    return request.getfixturevalue(request.param)


def test_check_variables_and_dataframe(df):
    # matches
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}
    es = EntitySet(id='test')
    es.entity_from_dataframe('test_entity', df, index='id',
                             variable_types=vtypes)
    assert es.entity_dict['test_entity'].variable_types['category'] == variable_types.Categorical


def test_make_index_variable_ordering(df):
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}

    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity',
                             index='id1',
                             make_index=True,
                             variable_types=vtypes,
                             dataframe=df)
    assert es.entity_dict['test_entity'].df.columns[0] == 'id1'


def test_extra_variable_type(df):
    # more variables
    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical,
              'category2': variable_types.Categorical}

    error_text = "Variable ID category2 not in DataFrame"
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id='test')
        es.entity_from_dataframe(entity_id='test_entity',
                                 index='id',
                                 variable_types=vtypes, dataframe=df)


def test_add_parent_not_index_varible(es):
    error_text = "Parent variable.*is not the index of entity Entity.*"
    with pytest.raises(AttributeError, match=error_text):
        es.add_relationship(Relationship(es[u'régions']['language'],
                                         es['customers'][u'région_id']))


@pytest.fixture
def pd_df2():
    return pd.DataFrame({'category': [1, 2, 3], 'category2': ['1', '2', '3']})


@pytest.fixture
def dd_df2(pd_df2):
    return dd.from_pandas(pd_df2, npartitions=2)


@pytest.fixture(params=['pd_df2', 'dd_df2'])
def df2(request):
    return request.getfixturevalue(request.param)


def test_none_index(df2):
    vtypes = {'category': variable_types.Categorical, 'category2': variable_types.Categorical}

    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity',
                             dataframe=df2,
                             variable_types=vtypes)
    assert es['test_entity'].index == 'category'
    assert isinstance(es['test_entity']['category'], variable_types.Index)


@pytest.fixture
def pd_df3():
    return pd.DataFrame({'category': [1, 2, 3]})


@pytest.fixture
def dd_df3(pd_df3):
    return dd.from_pandas(pd_df3, npartitions=2)


@pytest.fixture(params=['pd_df3', 'dd_df3'])
def df3(request):
    return request.getfixturevalue(request.param)


def test_unknown_index(df3):
    vtypes = {'category': variable_types.Categorical}

    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity',
                             index='id',
                             variable_types=vtypes, dataframe=df3)
    assert es['test_entity'].index == 'id'
    assert list(es['test_entity'].df['id']) == list(range(3))


def test_doesnt_remake_index(df):
    error_text = "Cannot make index: index variable already present"
    with pytest.raises(RuntimeError, match=error_text):
        es = EntitySet(id='test')
        es.entity_from_dataframe(entity_id='test_entity',
                                 index='id',
                                 make_index=True,
                                 dataframe=df)


def test_bad_time_index_variable(df3):
    error_text = "Time index not found in dataframe"
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id='test')
        es.entity_from_dataframe(entity_id='test_entity',
                                 index="id",
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


@pytest.fixture(params=['pd_df4', 'dd_df4'])
def df4(request):
    return request.getfixturevalue(request.param)


def test_converts_variable_types_on_init(df4):
    vtypes = {'id': variable_types.Categorical,
              'ints': variable_types.Numeric,
              'floats': variable_types.Numeric}
    if isinstance(df4, dd.DataFrame):
        vtypes['category'] = variable_types.Categorical
        vtypes['category_int'] = variable_types.Categorical
    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity', index='id',
                             variable_types=vtypes, dataframe=df4)

    entity_df = es['test_entity'].df
    assert entity_df['ints'].dtype.name in variable_types.PandasTypes._pandas_numerics
    assert entity_df['floats'].dtype.name in variable_types.PandasTypes._pandas_numerics

    # this is infer from pandas dtype
    e = es["test_entity"]
    assert isinstance(e['category_int'], variable_types.Categorical)


def test_converts_variable_type_after_init(df4):
    df4["category"] = df4["category"].astype("category")
    if isinstance(df4, dd.DataFrame):
        vtypes = {'id': variable_types.Categorical,
                  'category': variable_types.Categorical,
                  'category_int': variable_types.Categorical,
                  'ints': variable_types.Numeric,
                  'floats': variable_types.Numeric}
    else:
        vtypes = None
    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity', index='id',
                             dataframe=df4, variable_types=vtypes)
    e = es['test_entity']
    df = es['test_entity'].df

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


def test_errors_no_vtypes_dask(dd_df4):
    es = EntitySet(id='test')
    msg = 'Variable types cannot be inferred from Dask DataFrames, ' \
          'use variable_types to provide type metadata for entity'
    with pytest.raises(ValueError, match=msg):
        es.entity_from_dataframe(entity_id='test_entity', index='id',
                                 dataframe=dd_df4)


@pytest.fixture
def pd_datetime1():
    times = pd.date_range('1/1/2011', periods=3, freq='H')
    time_strs = times.strftime('%Y-%m-%d')
    return pd.DataFrame({'id': [0, 1, 2], 'time': time_strs})


@pytest.fixture
def dd_datetime1(pd_datetime1):
    return dd.from_pandas(pd_datetime1, npartitions=2)


@pytest.fixture(params=['pd_datetime1', 'dd_datetime1'])
def datetime1(request):
    return request.getfixturevalue(request.param)


def test_converts_datetime(datetime1):
    # string converts to datetime correctly
    # This test fails without defining vtypes.  Entityset
    # infers time column should be numeric type
    vtypes = {'id': variable_types.Categorical,
              'time': variable_types.Datetime}

    es = EntitySet(id='test')
    es.entity_from_dataframe(
        entity_id='test_entity',
        index='id',
        time_index="time",
        variable_types=vtypes,
        dataframe=datetime1)
    pd_col = es['test_entity'].df['time']
    if isinstance(pd_col, dd.Series):
        pd_col = pd_col.compute()
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


@pytest.fixture(params=['pd_datetime2', 'dd_datetime2'])
def datetime2(request):
    return request.getfixturevalue(request.param)


def test_handles_datetime_format(datetime2):
    # check if we load according to the format string
    # pass in an ambigious date
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp('Jan 2, 2011')

    vtypes = {'id': variable_types.Categorical,
              'time_format': (variable_types.Datetime, {"format": datetime_format}),
              'time_no_format': variable_types.Datetime}

    es = EntitySet(id='test')
    es.entity_from_dataframe(
        entity_id='test_entity',
        index='id',
        variable_types=vtypes,
        dataframe=datetime2)

    col_format = es['test_entity'].df['time_format']
    col_no_format = es['test_entity'].df['time_no_format']
    if isinstance(col_format, dd.Series):
        col_format = col_format.compute()
        col_no_format = col_no_format.compute()
    # without formatting pandas gets it wrong
    assert (col_no_format != actual).all()

    # with formatting we correctly get jan2
    assert (col_format == actual).all()


# Inferring variable types and verifying typing not supported in dask
def test_handles_datetime_mismatch():
    # can't convert arbitrary strings
    df = pd.DataFrame({'id': [0, 1, 2], 'time': ['a', 'b', 'tomorrow']})
    vtypes = {'id': variable_types.Categorical,
              'time': variable_types.Datetime}

    error_text = "Given date string not likely a datetime."
    with pytest.raises(ValueError, match=error_text):
        es = EntitySet(id='test')
        es.entity_from_dataframe('test_entity', df, 'id',
                                 time_index='time', variable_types=vtypes)


def test_entity_init(es):
    # Note: to convert the time column directly either the variable type
    # or convert_date_columns must be specifie
    df = pd.DataFrame({'id': [0, 1, 2],
                       'time': [datetime(2011, 4, 9, 10, 31, 3 * i)
                                for i in range(3)],
                       'category': ['a', 'b', 'a'],
                       'number': [4, 5, 6]})
    if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
        df = dd.from_pandas(df, npartitions=2)

    vtypes = {'time': variable_types.Datetime}
    if isinstance(df, dd.DataFrame):
        extra_vtypes = {
            'id': variable_types.Categorical,
            'category': variable_types.Categorical,
            'number': variable_types.Numeric
        }
        vtypes.update(extra_vtypes)
    es.entity_from_dataframe('test_entity', df, index='id',
                             time_index='time', variable_types=vtypes)
    if isinstance(df, dd.DataFrame):
        df_shape = (df.shape[0].compute(), df.shape[1])
    else:
        df_shape = df.shape
    if isinstance(es['test_entity'].df, dd.DataFrame):
        es_df_shape = (es['test_entity'].df.shape[0].compute(), es['test_entity'].df.shape[1])
    else:
        es_df_shape = es['test_entity'].df.shape
    assert es_df_shape == df_shape
    assert es['test_entity'].index == 'id'
    assert es['test_entity'].time_index == 'time'
    assert set([v.id for v in es['test_entity'].variables]) == set(df.columns)

    assert es['test_entity'].df['time'].dtype == df['time'].dtype
    assert set(es['test_entity'].df['id']) == set(df['id'])


@pytest.fixture
def pd_bad_df():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 3: ['a', 'b', 'c']})


@pytest.fixture
def dd_bad_df(pd_bad_df):
    return dd.from_pandas(pd_bad_df, npartitions=2)


@pytest.fixture(params=['pd_bad_df', 'dd_bad_df'])
def bad_df(request):
    return request.getfixturevalue(request.param)


def test_nonstr_column_names(bad_df):
    es = ft.EntitySet(id='Failure')
    error_text = r"All column names must be strings \(Column 3 is not a string\)"
    with pytest.raises(ValueError, match=error_text):
        es.entity_from_dataframe(entity_id='str_cols',
                                 dataframe=bad_df,
                                 index='index')


def test_sort_time_id():
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s")[::-1]})

    es = EntitySet("test", entities={"t": (transactions_df, "id", "transaction_time")})
    times = list(es["t"].df.transaction_time)
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
    es.entity_from_dataframe('t',
                             transactions_df,
                             index='id',
                             time_index="transaction_time",
                             already_sorted=True)
    times = list(es["t"].df.transaction_time)
    assert times == list(transactions_df.transaction_time)


# TODO: equality check fails, dask series have no .equals method; error computing lti if categorical index
def test_concat_entitysets(es):
    df = pd.DataFrame({'id': [0, 1, 2], 'category': ['a', 'b', 'a']})
    if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask has no .equals method and issue with categoricals "
                     "and add_last_time_indexes")
        df = dd.from_pandas(df, npartitions=2)

    vtypes = {'id': variable_types.Categorical,
              'category': variable_types.Categorical}
    es.entity_from_dataframe(entity_id='test_entity',
                             index='id1',
                             make_index=True,
                             variable_types=vtypes,
                             dataframe=df)
    es.add_last_time_indexes()

    assert es.__eq__(es)
    es_1 = copy.deepcopy(es)
    es_2 = copy.deepcopy(es)

    # map of what rows to take from es_1 and es_2 for each entity
    emap = {
        'log': [list(range(10)) + [14, 15, 16], list(range(10, 14)) + [15, 16]],
        'sessions': [[0, 1, 2, 5], [1, 3, 4, 5]],
        'customers': [[0, 2], [1, 2]],
        'test_entity': [[0, 1], [0, 2]],
    }

    assert es.__eq__(es_1, deep=True)
    assert es.__eq__(es_2, deep=True)

    for i, _es in enumerate([es_1, es_2]):
        for entity, rows in emap.items():
            df = _es[entity].df
            _es[entity].update_data(df=df.loc[rows[i]])

    assert 10 not in es_1['log'].last_time_index.index
    assert 10 in es_2['log'].last_time_index.index
    assert 9 in es_1['log'].last_time_index.index
    assert 9 not in es_2['log'].last_time_index.index
    assert not es.__eq__(es_1, deep=True)
    assert not es.__eq__(es_2, deep=True)

    # make sure internal indexes work before concat
    regions = es_1['customers'].query_by_values(['United States'], variable_id=u'région_id')
    assert regions.index.isin(es_1['customers'].df.index).all()

    assert es_1.__eq__(es_2)
    assert not es_1.__eq__(es_2, deep=True)

    old_es_1 = copy.deepcopy(es_1)
    old_es_2 = copy.deepcopy(es_2)
    es_3 = es_1.concat(es_2)

    assert old_es_1.__eq__(es_1, deep=True)
    assert old_es_2.__eq__(es_2, deep=True)

    assert es_3.__eq__(es)
    for entity in es.entities:
        df = es[entity.id].df.sort_index()
        df_3 = es_3[entity.id].df.sort_index()
        for column in df:
            for x, y in zip(df[column], df_3[column]):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
        orig_lti = es[entity.id].last_time_index.sort_index()
        new_lti = es_3[entity.id].last_time_index.sort_index()
        for x, y in zip(orig_lti, new_lti):
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

    es_1['stores'].last_time_index = None
    es_1['test_entity'].last_time_index = None
    es_2['test_entity'].last_time_index = None
    es_4 = es_1.concat(es_2)
    assert not es_4.__eq__(es, deep=True)
    for entity in es.entities:
        df = es[entity.id].df.sort_index()
        df_4 = es_4[entity.id].df.sort_index()
        for column in df:
            for x, y in zip(df[column], df_4[column]):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

        if entity.id != 'test_entity':
            orig_lti = es[entity.id].last_time_index.sort_index()
            new_lti = es_4[entity.id].last_time_index.sort_index()
            for x, y in zip(orig_lti, new_lti):
                assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
        else:
            assert es_4[entity.id].last_time_index is None


@pytest.fixture
def pd_transactions_df():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                         "card_id": [1, 2, 1, 3, 4, 5],
                         "transaction_time": [10, 12, 13, 20, 21, 20],
                         "fraud": [True, False, False, False, True, True]})


@pytest.fixture
def dd_transactions_df(pd_transactions_df):
    return dd.from_pandas(pd_transactions_df, npartitions=3)


@pytest.fixture(params=['pd_transactions_df', 'dd_transactions_df'])
def transactions_df(request):
    return request.getfixturevalue(request.param)


def test_set_time_type_on_init(transactions_df):
    # create cards entity
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    if isinstance(transactions_df, dd.DataFrame):
        cards_df = dd.from_pandas(cards_df, npartitions=3)
        cards_vtypes = {'id': variable_types.Categorical}
        transactions_vtypes = {
            'id': variable_types.Categorical,
            'card_id': variable_types.Categorical,
            'transaction_time': variable_types.Numeric,
            'fraud': variable_types.Boolean
        }
    else:
        cards_vtypes = None
        transactions_vtypes = None

    entities = {
        "cards": (cards_df, "id", None, cards_vtypes),
        "transactions": (transactions_df, "id", "transaction_time", transactions_vtypes)
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    es = EntitySet("fraud", entities, relationships)
    # assert time_type is set
    assert es.time_type == variable_types.NumericTimeIndex


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
        accounts_vtypes = {'id': variable_types.Categorical, 'signup_date': variable_types.Datetime}
        transactions_vtypes = {
            'id': variable_types.Categorical,
            'card_id': variable_types.Categorical,
            'transaction_time': variable_types.Numeric,
            'fraud': variable_types.Boolean
        }
    else:
        accounts_vtypes = None
        transactions_vtypes = None

    # create empty entityset
    es = EntitySet("fraud")
    # assert it's not set
    assert getattr(es, "time_type", None) is None
    # add entity
    es.entity_from_dataframe("transactions",
                             transactions_df,
                             index="id",
                             time_index="transaction_time",
                             variable_types=transactions_vtypes)
    # assert time_type is set
    assert es.time_type == variable_types.NumericTimeIndex
    # add another entity
    es.normalize_entity("transactions",
                        "cards",
                        "card_id",
                        make_time_index=True)
    # assert time_type unchanged
    assert es.time_type == variable_types.NumericTimeIndex
    # add wrong time type entity
    error_text = "accounts time index is <class 'featuretools.variable_types.variable.DatetimeTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.entity_from_dataframe("accounts",
                                 accounts_df,
                                 index="id",
                                 time_index="signup_date",
                                 variable_types=accounts_vtypes)
    # add non time type as time index, only valid for pandas
    if isinstance(transactions_df, pd.DataFrame):
        error_text = "Attempted to convert all string column signup_date to numeric"
        with pytest.raises(TypeError, match=error_text):
            es.entity_from_dataframe("accounts",
                                     accounts_df_string,
                                     index="id",
                                     time_index="signup_date")


def test_checks_time_type_setting_time_index(es):
    # set non time type as time index, Dask and Pandas error differently
    if isinstance(es['log'].df, pd.DataFrame):
        error_text = 'log time index not recognized as numeric or datetime'
    else:
        error_text = "log time index is %s type which differs from" \
                     " other entityset time indexes" % (variable_types.NumericTimeIndex)
    with pytest.raises(TypeError, match=error_text):
        es['log'].set_time_index('purchased')


def test_checks_time_type_setting_secondary_time_index(es):
    # entityset is timestamp time type
    assert es.time_type == variable_types.DatetimeTimeIndex
    # add secondary index that is timestamp type
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'cancel_date': ['cancel_date', 'cancel_reason']}
    es["customers"].set_secondary_time_index(new_2nd_ti)
    assert es.time_type == variable_types.DatetimeTimeIndex
    # add secondary index that is numeric type
    new_2nd_ti = {'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is <class 'featuretools.variable_types.variable.NumericTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es["customers"].set_secondary_time_index(new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {'favorite_quote': ['favorite_quote', 'loves_ice_cream']}

    error_text = r"data type (\"|')All members of the working classes must seize the means of production.(\"|') not understood"
    with pytest.raises(TypeError, match=error_text):
        es["customers"].set_secondary_time_index(new_2nd_ti)
    # add mismatched pair of secondary time indexes
    new_2nd_ti = {'upgrade_date': ['upgrade_date', 'favorite_quote'],
                  'age': ['age', 'loves_ice_cream']}

    error_text = "customers time index is <class 'featuretools.variable_types.variable.NumericTimeIndex'> type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es["customers"].set_secondary_time_index(new_2nd_ti)

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

    error_text = r"data type ('|\")City A('|\") not understood"
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


def test_normalize_entity(es):
    error_text = "'additional_variables' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_entity('sessions', 'device_types', 'device_type',
                            additional_variables='log')

    error_text = "'copy_variables' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_entity('sessions', 'device_types', 'device_type',
                            copy_variables='log')

    es.normalize_entity('sessions', 'device_types', 'device_type',
                        additional_variables=['device_name'],
                        make_time_index=False)

    assert len(es.get_forward_relationships('sessions')) == 2
    assert es.get_forward_relationships(
        'sessions')[1].parent_entity.id == 'device_types'
    assert 'device_name' in es['device_types'].df.columns
    assert 'device_name' not in es['sessions'].df.columns
    assert 'device_type' in es['device_types'].df.columns


def test_normalize_entity_new_time_index_in_base_entity_error_check(es):
    error_text = "'make_time_index' must be a variable in the base entity"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity(base_entity_id='customers',
                            new_entity_id='cancellations',
                            index='cancel_reason',
                            make_time_index="non-existent")


def test_normalize_entity_new_time_index_in_variable_list_error_check(es):
    error_text = "'make_time_index' must be specified in 'additional_variables' or 'copy_variables'"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity(base_entity_id='customers',
                            new_entity_id='cancellations',
                            index='cancel_reason',
                            make_time_index='cancel_date')


def test_normalize_entity_new_time_index_copy_success_check(es):
    es.normalize_entity(base_entity_id='customers',
                        new_entity_id='cancellations',
                        index='cancel_reason',
                        make_time_index='cancel_date',
                        additional_variables=[],
                        copy_variables=['cancel_date'])


def test_normalize_entity_new_time_index_additional_success_check(es):
    es.normalize_entity(base_entity_id='customers',
                        new_entity_id='cancellations',
                        index='cancel_reason',
                        make_time_index='cancel_date',
                        additional_variables=['cancel_date'],
                        copy_variables=[])


def test_normalize_time_index_from_none(es):
    es['customers'].time_index = None
    es.normalize_entity('customers', 'birthdays', 'age',
                        make_time_index='date_of_birth',
                        copy_variables=['date_of_birth'])
    assert es['birthdays'].time_index == 'date_of_birth'
    df = es['birthdays'].df

    # only pandas sorts by time index
    if isinstance(df, pd.DataFrame):
        assert df['date_of_birth'].is_monotonic_increasing


def test_raise_error_if_dupicate_additional_variables_passed(es):
    error_text = "'additional_variables' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity('sessions', 'device_types', 'device_type',
                            additional_variables=['device_name', 'device_name'])


def test_raise_error_if_dupicate_copy_variables_passed(es):
    error_text = "'copy_variables' contains duplicate variables. All variables must be unique."
    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity('sessions', 'device_types', 'device_type',
                            copy_variables=['device_name', 'device_name'])


def test_normalize_entity_copies_variable_types(es):
    es['log'].convert_variable_type(
        'value', variable_types.Ordinal, convert_data=False)
    assert es['log'].variable_types['value'] == variable_types.Ordinal
    assert es['log'].variable_types['priority_level'] == variable_types.Ordinal
    es.normalize_entity('log', 'values_2', 'value_2',
                        additional_variables=['priority_level'],
                        copy_variables=['value'],
                        make_time_index=False)

    assert len(es.get_forward_relationships('log')) == 3
    assert es.get_forward_relationships(
        'log')[2].parent_entity.id == 'values_2'
    assert 'priority_level' in es['values_2'].df.columns
    assert 'value' in es['values_2'].df.columns
    assert 'priority_level' not in es['log'].df.columns
    assert 'value' in es['log'].df.columns
    assert 'value_2' in es['values_2'].df.columns
    assert es['values_2'].variable_types['priority_level'] == variable_types.Ordinal
    assert es['values_2'].variable_types['value'] == variable_types.Ordinal


# sorting not supported in dask
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


def test_normalize_entity_new_time_index(es):
    new_time_index = 'value_time'
    es.normalize_entity('log', 'values', 'value',
                        make_time_index=True,
                        new_entity_time_index=new_time_index)

    assert es['values'].time_index == new_time_index
    assert new_time_index in es['values'].df.columns
    assert len(es['values'].df.columns) == 2
    df = es['values'].df
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    assert df[new_time_index].is_monotonic_increasing


def test_normalize_entity_same_index(es):
    transactions_df = pd.DataFrame({"id": [1, 2, 3],
                                    "transaction_time": pd.date_range(start="10:00", periods=3, freq="10s"),
                                    "first_entity_time": [1, 2, 3]})
    es = ft.EntitySet("example")
    es.entity_from_dataframe(entity_id="entity",
                             index="id",
                             time_index="transaction_time",
                             dataframe=transactions_df)

    error_text = "'index' must be different from the index column of the base entity"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity(base_entity_id="entity",
                            new_entity_id="new_entity",
                            index="id",
                            make_time_index=True)


# TODO: normalize entity fails with Dask, doesn't specify all vtypes when creating new entity
def test_secondary_time_index(es):
    if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
        pytest.xfail('vtype error when attempting to normalize entity')
    es.normalize_entity('log', 'values', 'value',
                        make_time_index=True,
                        make_secondary_time_index={
                               'datetime': ['comments']},
                        new_entity_time_index="value_time",
                        new_entity_secondary_time_index='second_ti')

    assert (isinstance(es['values'].df['second_ti'], pd.Series))
    assert (es['values']['second_ti'].type_string == 'datetime')
    assert (es['values'].secondary_time_index == {
            'second_ti': ['comments', 'second_ti']})


def test_sizeof(es):
    total_size = 0
    for entity in es.entities:
        total_size += entity.df.__sizeof__()
        total_size += entity.last_time_index.__sizeof__()

    assert es.__sizeof__() == total_size


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


@pytest.fixture
def pd_datetime3():
    return pd.DataFrame({'id': [0, 1, 2],
                         'ints': ['1', '2', '1']})


@pytest.fixture
def dd_datetime3(pd_datetime3):
    return dd.from_pandas(pd_datetime3, npartitions=2)


@pytest.fixture(params=['pd_datetime3', 'dd_datetime3'])
def datetime3(request):
    return request.getfixturevalue(request.param)


def test_datetime64_conversion(datetime3):
    df = datetime3
    df["time"] = pd.Timestamp.now()
    df["time"] = df["time"].astype("datetime64[ns, UTC]")

    if isinstance(df, dd.DataFrame):
        vtypes = {
            'id': variable_types.Categorical,
            'ints': variable_types.Numeric,
            'time': variable_types.Datetime
        }
    else:
        vtypes = None
    es = EntitySet(id='test')
    es.entity_from_dataframe(entity_id='test_entity',
                             index='id',
                             dataframe=df,
                             variable_types=vtypes)
    vtype_time_index = variable_types.variable.DatetimeTimeIndex
    es['test_entity'].convert_variable_type('time', vtype_time_index)


def test_later_schema_version(es):
    def test_version(major, minor, patch, raises=True):
        version = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved entityset'
                            '(%s) is greater than the latest supported (%s). '
                            'You may need to upgrade featuretools. Attempting to load entityset ...'
                            % (version, SCHEMA_VERSION))
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text)

    major, minor, patch = [int(s) for s in SCHEMA_VERSION.split('.')]

    test_version(major + 1, minor, patch)
    test_version(major, minor + 1, patch)
    test_version(major, minor, patch + 1)
    test_version(major, minor - 1, patch + 1, raises=False)


def test_earlier_schema_version(es):
    def test_version(major, minor, patch, raises=True):
        version = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved entityset'
                            '(%s) is no longer supported by this version '
                            'of featuretools. Attempting to load entityset ...'
                            % (version))
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text)

    major, minor, patch = [int(s) for s in SCHEMA_VERSION.split('.')]

    test_version(major - 1, minor, patch)
    test_version(major, minor - 1, patch, raises=False)
    test_version(major, minor, patch - 1, raises=False)


def _check_schema_version(version, es, warning_text):
    entities = {entity.id: serialize.entity_to_description(entity) for entity in es.entities}
    relationships = [relationship.to_dictionary() for relationship in es.relationships]
    dictionary = {
        'schema_version': version,
        'id': es.id,
        'entities': entities,
        'relationships': relationships,
    }

    if warning_text:
        with pytest.warns(UserWarning) as record:
            deserialize.description_to_entityset(dictionary)
        assert record[0].message.args[0] == warning_text
    else:
        deserialize.description_to_entityset(dictionary)


@pytest.fixture
def pd_index_df():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                         "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s"),
                         "first_entity_time": [1, 2, 3, 5, 6, 6]})


@pytest.fixture
def dd_index_df(pd_index_df):
    return dd.from_pandas(pd_index_df, npartitions=3)


@pytest.fixture(params=['pd_index_df', 'dd_index_df'])
def index_df(request):
    return request.getfixturevalue(request.param)


def test_same_index_values(index_df):
    if isinstance(index_df, dd.DataFrame):
        vtypes = {
            'id': variable_types.Categorical,
            'transaction_time': variable_types.Datetime,
            'first_entity_time': variable_types.Numeric
        }
    else:
        vtypes = None

    es = ft.EntitySet("example")

    error_text = "time_index and index cannot be the same value"
    with pytest.raises(ValueError, match=error_text):
        es.entity_from_dataframe(entity_id="entity",
                                 index="id",
                                 time_index="id",
                                 dataframe=index_df,
                                 variable_types=vtypes)

    es.entity_from_dataframe(entity_id="entity",
                             index="id",
                             time_index="transaction_time",
                             dataframe=index_df,
                             variable_types=vtypes)

    with pytest.raises(ValueError, match=error_text):
        es.normalize_entity(base_entity_id="entity",
                            new_entity_id="new_entity",
                            index="first_entity_time",
                            make_time_index=True)


def test_use_time_index(index_df):
    if isinstance(index_df, dd.DataFrame):
        bad_vtypes = {
            'id': variable_types.Categorical,
            'transaction_time': variable_types.DatetimeTimeIndex,
            'first_entity_time': variable_types.Numeric
        }
        vtypes = {
            'id': variable_types.Categorical,
            'transaction_time': variable_types.Datetime,
            'first_entity_time': variable_types.Numeric
        }
    else:
        bad_vtypes = {"transaction_time": variable_types.DatetimeTimeIndex}
        vtypes = None

    es = ft.EntitySet()

    error_text = "DatetimeTimeIndex variable transaction_time must be set using time_index parameter"
    with pytest.raises(ValueError, match=error_text):
        es.entity_from_dataframe(entity_id="entity",
                                 index="id",
                                 variable_types=bad_vtypes,
                                 dataframe=index_df)

    es.entity_from_dataframe(entity_id="entity",
                             index="id",
                             time_index="transaction_time",
                             variable_types=vtypes,
                             dataframe=index_df)


def test_normalize_with_datetime_time_index(es):
    es.normalize_entity(base_entity_id="customers",
                        new_entity_id="cancel_reason",
                        index="cancel_reason",
                        make_time_index=False,
                        copy_variables=['signup_date', 'upgrade_date'])

    vtypes = es['cancel_reason'].variable_types
    assert vtypes['signup_date'] == variable_types.Datetime
    assert vtypes['upgrade_date'] == variable_types.Datetime


def test_normalize_with_numeric_time_index(int_es):
    int_es.normalize_entity(base_entity_id="customers",
                            new_entity_id="cancel_reason",
                            index="cancel_reason",
                            make_time_index=False,
                            copy_variables=['signup_date', 'upgrade_date'])

    vtypes = int_es['cancel_reason'].variable_types
    assert vtypes['signup_date'] == variable_types.Numeric
    assert vtypes['upgrade_date'] == variable_types.Numeric


def test_normalize_with_invalid_time_index(es):
    es['customers'].convert_variable_type('signup_date', variable_types.Datetime)
    error_text = "Time index 'signup_date' is not a NumericTimeIndex or DatetimeTimeIndex," \
        + " but type <class 'featuretools.variable_types.variable.Datetime'>."\
        + " Use set_time_index on entity 'customers' to set the time_index."
    with pytest.raises(TypeError, match=error_text):
        es.normalize_entity(base_entity_id="customers",
                            new_entity_id="cancel_reason",
                            index="cancel_reason",
                            copy_variables=['upgrade_date'])
    es['customers'].convert_variable_type('signup_date', variable_types.DatetimeTimeIndex)


def test_entityset_init():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "upgrade_date": [51, 23, 45, 12, 22, 53],
                                    "fraud": [True, False, False, False, True, True]})
    variable_types = {
        'fraud': 'boolean',
        'card_id': 'categorical'
    }
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, 'id', 'transaction_time',
                         variable_types, False)
    }
    relationships = [('cards', 'id', 'transactions', 'card_id')]
    es = ft.EntitySet(id="fraud_data",
                      entities=entities,
                      relationships=relationships)
    assert es['transactions'].index == 'id'
    assert es['transactions'].time_index == 'transaction_time'
    es_copy = ft.EntitySet(id="fraud_data")
    es_copy.entity_from_dataframe(entity_id='cards',
                                  dataframe=cards_df,
                                  index='id')
    es_copy.entity_from_dataframe(entity_id='transactions',
                                  dataframe=transactions_df,
                                  index='id',
                                  variable_types=variable_types,
                                  make_index=False,
                                  time_index='transaction_time')
    relationship = ft.Relationship(es_copy["cards"]["id"],
                                   es_copy["transactions"]["card_id"])
    es_copy.add_relationship(relationship)
    assert es['cards'].__eq__(es_copy['cards'], deep=True)
    assert es['transactions'].__eq__(es_copy['transactions'], deep=True)
