import copy
import logging
from datetime import datetime

from woodwork.logical_types import Categorical, Integer
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
@pytest.fixture(params=['pd_df'])
def df(request):
    return request.getfixturevalue(request.param)


def test_init_es_with_dataframe(df):
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es.id == 'es'
    assert len(es.dataframe_dict) == 1
    assert es.dataframe_dict['table'] is df

    assert df.ww.schema is not None
    assert df.ww.logical_types['id'] == Integer
    assert df.ww.logical_types['category'] == Categorical


def test_init_es_with_woodwork_table(df):
    df.ww.init()
    es = EntitySet('es', dataframes={'table': (df,)})

    assert es.id == 'es'
    assert len(es.dataframe_dict) == 1
    assert es.dataframe_dict['table'] is df

    assert df.ww.schema is not None
    assert df.ww.logical_types['id'] == Integer
    assert df.ww.logical_types['category'] == Categorical


def test_init_es_with_dataframe_and_params(df):
    pass


def test_init_es_with_multiple_dataframes(df):
    pass


def test_add_dataframe_to_es(df):
    pass


def test_init_es_with_relationships(df):
    pass


def test_add_relationships_to_es(df):
    pass
