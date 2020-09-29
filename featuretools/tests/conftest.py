import copy
import sys

import composeml as cp
import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools.tests.testing_utils import (
    make_ecommerce_entityset,
    to_pandas
)
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean


@pytest.fixture(scope='session', autouse=True)
def spark_session():
    sql = import_or_none('pyspark.sql')
    if sql:
        spark = sql.SparkSession.builder \
            .master('local[2]') \
            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=True") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()

        return spark


@pytest.fixture(scope='session')
def make_es():
    return make_ecommerce_entityset()


@pytest.fixture(scope='session')
def make_int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


@pytest.fixture
def pd_es(make_es):
    return copy.deepcopy(make_es)


@pytest.fixture
def int_es(make_int_es):
    return copy.deepcopy(make_int_es)


@pytest.fixture
def dask_es(make_es):
    dask_es = copy.deepcopy(make_es)
    for entity in dask_es.entities:
        entity.df = dd.from_pandas(entity.df.reset_index(drop=True), npartitions=2)
    return dask_es


@pytest.fixture
def ks_es(make_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    ks_es = copy.deepcopy(make_es)
    for entity in ks_es.entities:
        cleaned_df = pd_to_ks_clean(entity.df).reset_index(drop=True)
        entity.df = ks.from_pandas(cleaned_df)
    return ks_es


@pytest.fixture(params=['pd_es', 'dask_es', 'ks_es'])
def es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=['pd_diamond_es', 'dask_diamond_es', 'ks_diamond_es'])
def diamond_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_diamond_es():
    countries_df = pd.DataFrame({
        'id': range(2),
        'name': ['US', 'Canada']
    })
    regions_df = pd.DataFrame({
        'id': range(3),
        'country_id': [0, 0, 1],
        'name': ['Northeast', 'South', 'Quebec'],
    })
    stores_df = pd.DataFrame({
        'id': range(5),
        'region_id': [0, 1, 2, 2, 1],
        'square_ft': [2000, 3000, 1500, 2500, 2700],
    })
    customers_df = pd.DataFrame({
        'id': range(5),
        'region_id': [1, 0, 0, 1, 1],
        'name': ['A', 'B', 'C', 'D', 'E'],
    })
    transactions_df = pd.DataFrame({
        'id': range(8),
        'store_id': [4, 4, 2, 3, 4, 0, 1, 1],
        'customer_id': [3, 0, 2, 4, 3, 3, 2, 3],
        'amount': [100, 40, 45, 83, 13, 94, 27, 81],
    })

    entities = {
        'countries': (countries_df, 'id'),
        'regions': (regions_df, 'id'),
        'stores': (stores_df, 'id'),
        'customers': (customers_df, 'id'),
        'transactions': (transactions_df, 'id'),
    }
    relationships = [
        ('countries', 'id', 'regions', 'country_id'),
        ('regions', 'id', 'stores', 'region_id'),
        ('regions', 'id', 'customers', 'region_id'),
        ('stores', 'id', 'transactions', 'store_id'),
        ('customers', 'id', 'transactions', 'customer_id'),
    ]
    return ft.EntitySet(id='ecommerce_diamond',
                        entities=entities,
                        relationships=relationships)


@pytest.fixture
def dask_diamond_es(pd_diamond_es):
    entities = {}
    for entity in pd_diamond_es.entities:
        entities[entity.id] = (dd.from_pandas(entity.df, npartitions=2), entity.index, None, entity.variable_types)

    relationships = [(rel.parent_entity.id,
                      rel.parent_variable.name,
                      rel.child_entity.id,
                      rel.child_variable.name) for rel in pd_diamond_es.relationships]

    return ft.EntitySet(id=pd_diamond_es.id, entities=entities, relationships=relationships)


@pytest.fixture
def ks_diamond_es(pd_diamond_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    entities = {}
    for entity in pd_diamond_es.entities:
        entities[entity.id] = (ks.from_pandas(pd_to_ks_clean(entity.df)), entity.index, None, entity.variable_types)

    relationships = [(rel.parent_entity.id,
                      rel.parent_variable.name,
                      rel.child_entity.id,
                      rel.child_variable.name) for rel in pd_diamond_es.relationships]

    return ft.EntitySet(id=pd_diamond_es.id, entities=entities, relationships=relationships)


@pytest.fixture(params=['pd_home_games_es', 'dask_home_games_es', 'ks_home_games_es'])
def home_games_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_home_games_es():
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
    entities = {'teams': (teams, 'id'), 'games': (games, 'id')}
    relationships = [('teams', 'id', 'games', 'home_team_id')]
    return ft.EntitySet(entities=entities,
                        relationships=relationships)


@pytest.fixture
def dask_home_games_es(pd_home_games_es):
    entities = {}
    for entity in pd_home_games_es.entities:
        entities[entity.id] = (dd.from_pandas(entity.df, npartitions=2), entity.index, None, entity.variable_types)

    relationships = [(rel.parent_entity.id,
                      rel.parent_variable.name,
                      rel.child_entity.id,
                      rel.child_variable.name) for rel in pd_home_games_es.relationships]

    return ft.EntitySet(id=pd_home_games_es.id, entities=entities, relationships=relationships)


@pytest.fixture
def ks_home_games_es(pd_home_games_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    entities = {}
    for entity in pd_home_games_es.entities:
        entities[entity.id] = (ks.from_pandas(pd_to_ks_clean(entity.df)), entity.index, None, entity.variable_types)

    relationships = [(rel.parent_entity.id,
                      rel.parent_variable.name,
                      rel.child_entity.id,
                      rel.child_variable.name) for rel in pd_home_games_es.relationships]

    return ft.EntitySet(id=pd_home_games_es.id, entities=entities, relationships=relationships)


@pytest.fixture
def games_es(home_games_es):
    away_team = ft.Relationship(home_games_es['teams']['id'],
                                home_games_es['games']['away_team_id'])
    return home_games_es.add_relationship(away_team)


@pytest.fixture
def pd_mock_customer():
    return ft.demo.load_mock_customer(return_entityset=True, random_seed=0)


@pytest.fixture
def dd_mock_customer(pd_mock_customer):
    dd_mock_customer = copy.deepcopy(pd_mock_customer)
    for entity in dd_mock_customer.entities:
        entity.df = dd.from_pandas(entity.df.reset_index(drop=True), npartitions=4)
    return dd_mock_customer


@pytest.fixture
def ks_mock_customer(pd_mock_customer):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    ks_mock_customer = copy.deepcopy(pd_mock_customer)
    for entity in ks_mock_customer.entities:
        cleaned_df = pd_to_ks_clean(entity.df).reset_index(drop=True)
        entity.df = ks.from_pandas(cleaned_df)
    return ks_mock_customer


@pytest.fixture(params=['pd_mock_customer', 'dd_mock_customer', 'ks_mock_customer'])
def mock_customer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def lt(es):
    def label_func(df):
        return df['value'].sum() > 10

    lm = cp.LabelMaker(
        target_entity='id',
        time_index='datetime',
        labeling_function=label_func,
        window_size='1m'
    )

    df = es['log'].df
    df = to_pandas(df)
    labels = lm.search(
        df,
        num_examples_per_instance=-1
    )
    labels = labels.rename(columns={'cutoff_time': 'time'})
    return labels
