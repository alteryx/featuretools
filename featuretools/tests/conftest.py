# import copy

import composeml as cp
import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools import variable_types as vtypes
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
            .config("spark.driver.bindAddress", "127.0.0.1") \
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
    # --> TODO temporary while waiting to implement deepcopy
    return make_ecommerce_entityset()


@pytest.fixture
def int_es(make_int_es):
    # --> TODO temporary while waiting to implement deepcopy
    return make_ecommerce_entityset(with_integer_time_index=True)


@pytest.fixture
def dask_es(make_es):
    es = ft.EntitySet(id=make_es.id)
    for df in make_es.dataframes:
        dd_df = dd.from_pandas(df.reset_index(drop=True), npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(df.ww.name, dd_df)

    for rel in make_es.relationships:
        es.add_relationship(rel.parent_dataframe.ww.name, rel.parent_column.name,
                            rel.child_dataframe.ww.name, rel.child_column.name)
    return es


@pytest.fixture
def ks_es(make_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    es = ft.EntitySet(id=make_es.id)
    for df in make_es.dataframes:
        cleaned_df = pd_to_ks_clean(df).reset_index(drop=True)
        ks_df = ks.from_pandas(cleaned_df)
        ks_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(df.ww.name, ks_df)

    for rel in make_es.relationships:
        es.add_relationship(rel._parent_dataframe_id, rel._parent_column_id,
                            rel._child_dataframe_id, rel._child_column_id)
    return es


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

    dataframes = {
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
                        dataframes=dataframes,
                        relationships=relationships)


@pytest.fixture
def dask_diamond_es(pd_diamond_es):
    dataframes = {}
    for df in pd_diamond_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=2)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_diamond_es.relationships]

    return ft.EntitySet(id=pd_diamond_es.id, dataframes=dataframes, relationships=relationships)


@pytest.fixture
def ks_diamond_es(pd_diamond_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    dataframes = {}
    for df in pd_diamond_es.dataframes:
        ks_df = ks.from_pandas(pd_to_ks_clean(df))
        ks_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (ks_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_diamond_es.relationships]

    return ft.EntitySet(id=pd_diamond_es.id, dataframes=dataframes, relationships=relationships)


@pytest.fixture(params=['pd_default_value_es', 'dask_default_value_es', 'ks_default_value_es'])
def default_value_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_default_value_es():
    transactions = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "session_id": ["a", "a", "b", "c"],
        "value": [1, 1, 1, 1]
    })

    sessions = pd.DataFrame({
        "id": ["a", "b"]
    })

    es = ft.EntitySet()
    es.entity_from_dataframe(entity_id="transactions",
                             dataframe=transactions,
                             index="id")
    es.entity_from_dataframe(entity_id="sessions",
                             dataframe=sessions,
                             index="id")

    es.add_relationship("sessions", "id", "transactions", "session_id")
    return es


@pytest.fixture
def dask_default_value_es(pd_default_value_es):
    dataframes = {}
    for df in pd_default_value_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_default_value_es.relationships]

    return ft.EntitySet(id=pd_default_value_es.id, dataframes=dataframes, relationships=relationships)


@pytest.fixture
def ks_default_value_es(pd_default_value_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    dataframes = {}
    for df in pd_default_value_es.dataframes:
        ks_df = ks.from_pandas(pd_to_ks_clean(df))
        ks_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (ks_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_default_value_es.relationships]

    return ft.EntitySet(id=pd_default_value_es.id, dataframes=dataframes, relationships=relationships)


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
    dataframes = {'teams': (teams, 'id'), 'games': (games, 'id')}
    relationships = [('teams', 'id', 'games', 'home_team_id')]
    return ft.EntitySet(dataframes=dataframes,
                        relationships=relationships)


@pytest.fixture
def dask_home_games_es(pd_home_games_es):
    dataframes = {}
    for df in pd_home_games_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=2)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_home_games_es.relationships]

    return ft.EntitySet(id=pd_home_games_es.id, dataframes=dataframes, relationships=relationships)


@pytest.fixture
def ks_home_games_es(pd_home_games_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    dataframes = {}
    for df in pd_home_games_es.dataframes:
        ks_df = ks.from_pandas(pd_to_ks_clean(df))
        ks_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (ks_df,)

    relationships = [(rel._parent_dataframe_id,
                      rel._parent_column_id,
                      rel._child_dataframe_id,
                      rel._child_column_id) for rel in pd_home_games_es.relationships]

    return ft.EntitySet(id=pd_home_games_es.id, dataframes=dataframes, relationships=relationships)


@pytest.fixture
def games_es(home_games_es):
    return home_games_es.add_relationship('teams', 'id', 'games', 'away_team_id')


@pytest.fixture
def pd_mock_customer():
    return ft.demo.load_mock_customer(return_entityset=True, random_seed=0)


@pytest.fixture
def dd_mock_customer(pd_mock_customer):
    entities = {}
    for entity in pd_mock_customer.entities:
        entities[entity.id] = (dd.from_pandas(entity.df.reset_index(drop=True), npartitions=4),
                               entity.index,
                               entity.time_index,
                               entity.variable_types)

    relationships = [(rel.parent_dataframe.id,
                      rel.parent_column.name,
                      rel.child_dataframe.id,
                      rel.child_column.name) for rel in pd_mock_customer.relationships]

    return ft.EntitySet(id=pd_mock_customer.id, entities=entities, relationships=relationships)


@pytest.fixture
def ks_mock_customer(pd_mock_customer):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    entities = {}
    for entity in pd_mock_customer.entities:
        cleaned_df = pd_to_ks_clean(entity.df).reset_index(drop=True)
        entities[entity.id] = (ks.from_pandas(cleaned_df),
                               entity.index,
                               entity.time_index,
                               entity.variable_types)

    relationships = [(rel.parent_dataframe.id,
                      rel.parent_column.name,
                      rel.child_dataframe.id,
                      rel.child_column.name) for rel in pd_mock_customer.relationships]

    return ft.EntitySet(id=pd_mock_customer.id, entities=entities, relationships=relationships)


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


@pytest.fixture(params=['pd_entities', 'dask_entities', 'koalas_entities'])
def entities(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_entities():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    return entities


@pytest.fixture
def dask_entities():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    cards_df = dd.from_pandas(cards_df, npartitions=2)
    transactions_df = dd.from_pandas(transactions_df, npartitions=2)

    cards_vtypes = {
        'id': vtypes.Index
    }
    transactions_vtypes = {
        'id': vtypes.Index,
        'card_id': vtypes.Id,
        'transaction_time': vtypes.NumericTimeIndex,
        'fraud': vtypes.Boolean
    }

    entities = {
        "cards": (cards_df, "id", None, cards_vtypes),
        "transactions": (transactions_df, "id", "transaction_time", transactions_vtypes)
    }
    return entities


@pytest.fixture
def koalas_entities():
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    cards_df = ks.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = ks.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    cards_vtypes = {
        'id': vtypes.Index
    }
    transactions_vtypes = {
        'id': vtypes.Index,
        'card_id': vtypes.Id,
        'transaction_time': vtypes.NumericTimeIndex,
        'fraud': vtypes.Boolean
    }

    entities = {
        "cards": (cards_df, "id", None, cards_vtypes),
        "transactions": (transactions_df, "id", "transaction_time", transactions_vtypes)
    }
    return entities


@pytest.fixture
def relationships():
    return [("cards", "id", "transactions", "card_id")]
