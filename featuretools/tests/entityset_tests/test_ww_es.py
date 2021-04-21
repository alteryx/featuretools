import copy
import logging
from datetime import datetime

from woodwork.logical_types import Categorical, Integer, NaturalLanguage
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools import variable_types
from featuretools.entityset.ww_entityset import EntitySet
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean


def test_empty_es():
    es = EntitySet('es')
    assert es.id == 'es'
    assert es.dataframe_dict == {}
    assert es.relationships == []
    assert es.time_type == None


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


# --> add other fixtures back in
@pytest.fixture(params=['pd_df', 'dd_df'])
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

    assert es['table'].ww.index is None
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
    second_df.ww.init(name='second_table', index='id', semantic_tags={'first_table_id': 'foreign_key'})

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


def test_init_es_with_relationships(df):
    pass


def test_add_relationships_to_es(df):
    pass
