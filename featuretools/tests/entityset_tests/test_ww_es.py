from datetime import datetime
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import Categorical, Integer, NaturalLanguage, Datetime
import woodwork as ww

from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none

ks = import_or_none('databricks.koalas')


def test_empty_es():
    es = EntitySet('es')
    assert es.id == 'es'
    assert es.dataframe_dict == {}
    assert es.relationships == []
    assert es.time_type is None


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


def test_init_es_with_dataframe(df):
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es.id == 'es'
    assert len(es.dataframe_dict) == 1
    assert es['table'] is df

    assert es['table'].ww.schema is not None
    assert es['table'].ww.logical_types['id'] == Integer
    assert es['table'].ww.logical_types['category'] == Categorical


def test_init_es_with_woodwork_table(df):
    df.ww.init(index='id')
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es.id == 'es'
    assert len(es.dataframe_dict) == 1
    assert es['table'] is df

    assert es['table'].ww.schema is not None

    assert es['table'].ww.index == 'id'
    assert es['table'].ww.time_index is None

    assert es['table'].ww.logical_types['id'] == Integer
    assert es['table'].ww.logical_types['category'] == Categorical


def test_init_es_with_dataframe_and_params(df):
    logical_types = {'id': 'NaturalLanguage', 'category': NaturalLanguage}
    semantic_tags = {'category': 'new_tag'}
    es = EntitySet('es', dataframes={'table': (df, 'id', None, logical_types, semantic_tags)})

    assert es.id == 'es'
    assert len(es.dataframe_dict) == 1
    assert es['table'] is df

    assert es['table'].ww.schema is not None

    assert es['table'].ww.index == 'id'
    assert es['table'].ww.time_index is None

    assert es['table'].ww.logical_types['id'] == NaturalLanguage
    assert es['table'].ww.logical_types['category'] == NaturalLanguage

    assert es['table'].ww.semantic_tags['id'] == {'index'}
    assert es['table'].ww.semantic_tags['category'] == {'new_tag'}


def test_init_es_with_multiple_dataframes(pd_df):
    second_df = pd.DataFrame({'id': [0, 1, 2, 3], 'first_table_id': [1, 2, 2, 1]})

    pd_df.ww.init(name='first_table', index='id')

    es = EntitySet('es', dataframes={'first_table': (pd_df,), 'second_table': (second_df, 'id', None, None, {'first_table_id': 'foreign_key'})})

    assert len(es.dataframe_dict) == 2
    assert es['first_table'].ww.schema is not None
    assert es['second_table'].ww.schema is not None


def test_add_dataframe_to_es(df):
    es1 = EntitySet('es')
    assert es1.dataframe_dict == {}
    es1.add_dataframe('table', df, index='id', semantic_tags={'category': 'new_tag'})
    assert len(es1.dataframe_dict) == 1

    copy_df = df.ww.copy()

    es2 = EntitySet('es')
    assert es2.dataframe_dict == {}
    es2.add_dataframe('table', copy_df)
    assert len(es2.dataframe_dict) == 1

    assert es1['table'].ww == es2['table'].ww


def test_change_es_dataframe_schema(df):
    df.ww.init(index='id')
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es['table'].ww.index == 'id'

    es.dataframe_dict['table'].ww.set_index('category')
    assert es['table'].ww.index == 'category'


def test_init_es_with_relationships(pd_df):
    second_df = pd.DataFrame({'id': [0, 1, 2, 3], 'first_table_id': [1, 2, 2, 1]})

    pd_df.ww.init(name='first_table', index='id')
    second_df.ww.init(name='second_table', index='id', semantic_tags={'first_table_id': 'foreign_key'})

    es = EntitySet('es',
                   dataframes={'first_table': (pd_df,), 'second_table': (second_df,)},
                   relationships=[('first_table', 'id', 'second_table', 'first_table_id')])

    assert len(es.relationships) == 1

    forward_dataframes = [name for name, _ in es.get_forward_dataframes('second_table')]
    assert forward_dataframes[0] == 'first_table'


def test_add_secondary_time_index():
    df = pd.DataFrame({
        'backwards_order': [8, 7, 6, 5, 4, 3, 2, 1, 0],
        'dates_backwards': ['2020-09-09', '2020-09-08', '2020-09-07', '2020-09-06', '2020-09-05', '2020-09-04', '2020-09-03', '2020-09-02', '2020-09-01'],
        'random_order': [7, 6, 8, 0, 2, 4, 3, 1, 5],
        'repeating_dates': ['2020-08-01', '2019-08-01', '2020-08-01', '2012-08-01', '2019-08-01', '2019-08-01', '2019-08-01', '2013-08-01', '2019-08-01'],
        'special': [7, 8, 0, 1, 4, 2, 6, 3, 5],
        'special_dates': ['2020-08-01', '2019-08-01', '2020-08-01', '2012-08-01', '2019-08-01', '2019-08-01', '2019-08-01', '2013-08-01', '2019-08-01'],
    })
    df.ww.init(index='backwards_order', time_index='dates_backwards')
    es = EntitySet('es')
    es.add_dataframe('dates_table', df, secondary_time_index={'repeating_dates': ['random_order', 'special']})

    assert df.ww.metadata['secondary_time_index'] == {'repeating_dates': ['random_order', 'special', 'repeating_dates']}


def test_normalize_dataframe():
    df = pd.DataFrame({
        'id': range(4),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com', 'junk@example.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555', '555-555-5555'],
        'age': pd.Series([33, None, 33, 57], dtype='Int64'),
        'signup_date': [pd.to_datetime('2020-09-01')] * 4,
        'is_registered': pd.Series([True, False, True, None], dtype='boolean'),
    })

    df.ww.init(index='id', time_index='signup_date')
    es = EntitySet('es')
    es.add_dataframe('first_table', df)
    es.normalize_dataframe('first_table', 'second_table', 'age',
                           additional_columns=['phone_number', 'full_name'],
                           make_time_index=True)
    assert len(es.dataframe_dict) == 2
    assert 'foreign_key' in es['first_table'].ww.semantic_tags['age']


def test_update_dataframe():
    df = pd.DataFrame({
        'id': range(4),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com', 'junk@example.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555', '555-555-5555'],
        'age': pd.Series([33, None, 33, 57], dtype='Int64'),
        'signup_date': [pd.to_datetime('2020-09-01')] * 4,
        'is_registered': pd.Series([True, False, True, None], dtype='boolean'),
    })

    df.ww.init(index='id')
    es = EntitySet('es')
    es.add_dataframe('table', df)
    original_schema = es['table'].ww.schema

    new_df = df.iloc[2:]
    es.update_dataframe('table', new_df)

    assert len(es['table']) == 2
    assert es['table'].ww.schema == original_schema


def test_add_last_time_indexes(es):
    es.add_last_time_indexes(['products'])

    assert 'last_time_index' in es['products'].ww.metadata


def test_conflicting_dataframe_names(es):
    new_es = EntitySet()

    sessions_df = es['sessions'].ww.copy()

    assert sessions_df.ww.name == 'sessions'

    new_es.add_dataframe('different_name', sessions_df)
    assert sessions_df.ww.name == 'different_name'
    assert new_es['different_name'] is sessions_df
    assert 'sessions' not in new_es.dataframe_dict


def test_extra_woodwork_params(es):
    new_es = EntitySet()

    sessions_df = es['sessions'].ww.copy()

    assert sessions_df.ww.index == 'id'
    assert sessions_df.ww.time_index is None
    assert sessions_df.ww.logical_types['id'] == Integer

    warning_msg = ('A Woodwork-initialized DataFrame was provided, so the following parameters were ignored: '
                   'index, make_index, time_index, logical_types, semantic_tags, already_sorted')
    with pytest.warns(UserWarning, match=warning_msg):
        new_es.add_dataframe(dataframe_id='sessions', dataframe=sessions_df,
                             index='filepath', time_index='customer_id',
                             logical_types={'id': Categorical}, make_index=True,
                             already_sorted=True, semantic_tags={'id': 'new_tag'})
    assert sessions_df.ww.index == 'id'
    assert sessions_df.ww.time_index is None
    assert sessions_df.ww.logical_types['id'] == Integer
    assert 'new_tag' not in sessions_df.ww.semantic_tags


def test_update_dataframe_errors(es):
    df = es['customers'].copy()
    if ks and isinstance(df, ks.DataFrame):
        df['new'] = [1, 2, 3]
    else:
        df['new'] = pd.Series([1, 2, 3])

    error_text = 'Updated dataframe is missing new cohort column'
    with pytest.raises(ValueError, match=error_text):
        es.update_dataframe(dataframe_id='customers', df=df.drop(columns=['cohort']))

    error_text = 'Updated dataframe contains 16 columns, expecting 15'
    with pytest.raises(ValueError, match=error_text):
        es.update_dataframe(dataframe_id='customers', df=df)


def test_update_dataframe_already_sorted(es):
    # test already_sorted on entity without time index
    df = es["sessions"].copy()
    updated_id = to_pandas(df['id'])
    updated_id.iloc[1] = 2
    updated_id.iloc[2] = 1

    df = df.set_index('id', drop=False)
    df.index.name = None

    assert es["sessions"].ww.time_index is None

    if ks and isinstance(df, ks.DataFrame):
        df["id"] = updated_id.to_list()
        df = df.sort_index()
    elif dd and isinstance(df, dd.DataFrame):
        df["id"] = updated_id
    else:
        assert df['id'].iloc[1] == 2

    es.update_dataframe(dataframe_id='sessions', df=df.copy(), already_sorted=False)
    sessions_df = to_pandas(es['sessions'])
    assert sessions_df["id"].iloc[1] == 2  # no sorting since time index not defined
    es.update_dataframe(dataframe_id='sessions', df=df.copy(), already_sorted=True)
    sessions_df = to_pandas(es['sessions'])
    assert sessions_df["id"].iloc[1] == 2

    # test already_sorted on entity with time index
    df = es["customers"].copy()
    updated_signup = to_pandas(df['signup_date'])
    updated_signup.iloc[0] = datetime(2011, 4, 11)

    assert es["customers"].ww.time_index == 'signup_date'

    if ks and isinstance(df, ks.DataFrame):
        df['signup_date'] = updated_signup.to_list()
        df = df.sort_index()
    else:
        df['signup_date'] = updated_signup

    es.update_dataframe(dataframe_id='customers', df=df.copy(), already_sorted=True)
    customers_df = to_pandas(es['customers'])
    assert customers_df["id"].iloc[0] == 2

    # only pandas allows for sorting:
    es.update_dataframe(dataframe_id='customers', df=df.copy(), already_sorted=False)
    updated_customers = to_pandas(es['customers'])
    if isinstance(df, pd.DataFrame):
        assert updated_customers["id"].iloc[0] == 0
    else:
        assert updated_customers["id"].iloc[0] == 2


def test_update_dataframe_invalid_schema(es):
    if not isinstance(es['customers'], pd.DataFrame):
        pytest.xfail('Invalid schema checks able to be caught by Woodwork only relevant for Pandas')
    df = es['customers'].copy()
    df['id'] = pd.Series([1, 1, 1])

    error_text = 'Woodwork typing information is not valid for this DataFrame: Index mismatch between DataFrame and typing information'
    with pytest.raises(ValueError, match=error_text):
        es.update_dataframe(dataframe_id='customers', df=df)


def test_update_dataframe_different_dtypes(es):
    float_dtype_df = es['customers'].copy()
    float_dtype_df = float_dtype_df.astype({'age': 'float64'})

    es.update_dataframe(dataframe_id='customers', df=float_dtype_df)

    assert es['customers']['age'].dtype == 'int64'
    assert es['customers'].ww.logical_types['age'] == Integer

    incompatible_dtype_df = es['customers'].copy()
    incompatible_list = ['hi', 'bye', 'bye']
    if ks and isinstance(incompatible_dtype_df, ks.DataFrame):
        incompatible_dtype_df['age'] = incompatible_list
    else:
        incompatible_dtype_df['age'] = pd.Series(incompatible_list)

    if isinstance(es['customers'], pd.DataFrame):
        # Dask and Koalas do not error on invalid type conversion until compute
        error_msg = 'Error converting datatype for age from type object to type int64. Please confirm the underlying data is consistent with logical type Integer.'
        with pytest.raises(ww.exceptions.TypeConversionError, match=error_msg):
            es.update_dataframe(dataframe_id='customers', df=incompatible_dtype_df)


@pytest.fixture()
def latlong_df_pandas():
    latlong_df = pd.DataFrame({
        'string_tuple': pd.Series(['(1, 2)', '(3, 4)']),
        'bracketless_string_tuple': pd.Series(['1, 2', '3, 4']),
        'list_strings': pd.Series([['1', '2'], ['3', '4']]),
        'combo_tuple_types': pd.Series(['[1, 2]', '(3, 4)']),
    })
    latlong_df.set_index('string_tuple', drop=False, inplace=True)
    latlong_df.index.name = None
    return latlong_df


@pytest.fixture()
def latlong_df_dask(latlong_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(latlong_df_pandas, npartitions=2)


@pytest.fixture()
def latlong_df_koalas(latlong_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(latlong_df_pandas.applymap(lambda tup: list(tup) if isinstance(tup, tuple) else tup))


@pytest.fixture(params=['latlong_df_pandas', 'latlong_df_dask', 'latlong_df_koalas'])
def latlong_df(request):
    return request.getfixturevalue(request.param)


def test_update_dataframe_data_transformation(latlong_df):
    initial_df = latlong_df.copy()
    initial_df.ww.init(index='string_tuple', logical_types={col_name: 'LatLong' for col_name in initial_df.columns})
    es = EntitySet()
    es.add_dataframe(dataframe_id='latlongs', dataframe=initial_df)

    df = to_pandas(es['latlongs'])
    expected_val = (1, 2)
    if ks and isinstance(es['latlongs'], ks.DataFrame):
        expected_val = [1, 2]
    for col in latlong_df.columns:
        series = df[col]
        assert series.iloc[0] == expected_val

    latlong_df.index = initial_df.index  # Need the underlying index to match
    es.update_dataframe('latlongs', latlong_df)
    df = to_pandas(es['latlongs'])
    expected_val = (3, 4)
    if ks and isinstance(es['latlongs'], ks.DataFrame):
        expected_val = [3, 4]
    for col in latlong_df.columns:
        series = df[col]
        assert series.iloc[-1] == expected_val


def test_update_dataframe_column_order(es):
    original_column_order = es['customers'].columns.copy()

    df = es['customers'].copy()
    col = df.pop('cohort')
    df[col.name] = col

    assert not df.columns.equals(original_column_order)
    assert set(df.columns) == set(original_column_order)

    es.update_dataframe(dataframe_id='customers', df=df)

    assert es['customers'].columns.equals(original_column_order)


def test_update_dataframe_woodwork_initialized(es):
    df = es['customers'].copy()
    if ks and isinstance(df, ks.DataFrame):
        df['age'] = [1, 2, 3]
    else:
        df['age'] = pd.Series([1, 2, 3])

    df.ww.init(schema=es['customers'].ww.schema)

    # Change the original Schema
    es['customers'].ww.metadata['user'] = 'user0'
    original_schema = es['customers'].ww.schema
    assert 'user' in es['customers'].ww.metadata

    es.update_dataframe('customers', df, already_sorted=True)

    if dd and isinstance(df, dd.DataFrame):
        assert all(to_pandas(es['customers']['age']) == [1, 2, 3])
    else:
        assert all(to_pandas(es['customers']['age']) == [3, 1, 2])

    assert es['customers'].ww.schema != original_schema
    assert 'user' not in es['customers'].ww.metadata


def test_update_dataframe_different_dataframe_types():
    dask_es = EntitySet(id="dask_es")

    sessions = pd.DataFrame({"id": [0, 1, 2, 3],
                             "user": [1, 2, 1, 3],
                             "time": [pd.to_datetime('2019-01-10'),
                                      pd.to_datetime('2019-02-03'),
                                      pd.to_datetime('2019-01-01'),
                                      pd.to_datetime('2017-08-25')],
                             "strings": ["I am a string",
                                         "23",
                                         "abcdef ghijk",
                                         ""]})
    sessions_dask = dd.from_pandas(sessions, npartitions=2)
    sessions_logical_types = {
        "id": Integer,
        "user": Integer,
        "time": Datetime,
        "strings": NaturalLanguage
    }
    sessions_semantic_tags = {'user': 'foreign_key'}

    dask_es.add_dataframe(dataframe_id="sessions", dataframe=sessions_dask, index="id", time_index="time",
                          logical_types=sessions_logical_types, semantic_tags=sessions_semantic_tags)

    with pytest.raises(TypeError, match='Incorrect DataFrame type used'):
        dask_es.update_dataframe('sessions', sessions)


def test_update_dataframe_last_time_index(es):
    # --> koalas currently fails bc we can't get the schema because it deepcopies
    es.add_last_time_indexes()
    df = es['customers'].copy()
    original_last_time_index = to_pandas(es['customers'].ww.metadata['last_time_index'].copy())

    new_time_index = ['2012-04-06', '2012-04-08', '2012-04-09']
    if ks and isinstance(df, ks.DataFrame):
        df['signup_date'] = new_time_index
    else:
        df['signup_date'] = pd.Series(new_time_index)

    es.update_dataframe('customers', df, recalculate_last_time_indexes=False)
    assert original_last_time_index.equals(to_pandas(es['customers'].ww.metadata['last_time_index']))

    es.update_dataframe('customers', df, recalculate_last_time_indexes=True)
    assert not original_last_time_index.equals(to_pandas(es['customers'].ww.metadata['last_time_index']))
