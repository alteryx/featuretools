from datetime import datetime
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import Categorical, Integer, NaturalLanguage

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
    df.ww.init()
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
    df.ww.init()
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es['table'].ww.index == 'id'

    es.dataframe_dict['table'].ww.set_index('id')
    assert es['table'].ww.index == 'id'


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
    # --> test with ww initalized dataframe, test with dtype change, test with other index problems
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
    pass


def test_update_dataframe_different_dtypes(es):
    pass
