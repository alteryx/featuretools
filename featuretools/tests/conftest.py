import copy

import composeml as cp
import dask.dataframe as dd
import pandas as pd
import pytest
from distributed import LocalCluster
from woodwork.logical_types import Boolean, Integer

from featuretools import EntitySet, demo
from featuretools.tests.testing_utils import make_ecommerce_entityset, to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.spark_utils import pd_to_spark_clean


@pytest.fixture()
def dask_cluster():
    with LocalCluster() as cluster:
        yield cluster


@pytest.fixture()
def three_worker_dask_cluster():
    with LocalCluster(n_workers=3) as cluster:
        yield cluster


@pytest.fixture(scope="session", autouse=True)
def spark_session():
    sql = import_or_none("pyspark.sql")
    if sql:
        spark = (
            sql.SparkSession.builder.master("local[2]")
            .config(
                "spark.driver.extraJavaOptions",
                "-Dio.netty.tryReflectionSetAccessible=True",
            )
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

        return spark


@pytest.fixture(scope="session")
def make_es():
    return make_ecommerce_entityset()


@pytest.fixture(scope="session")
def make_int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


@pytest.fixture
def pd_es(make_es):
    return copy.deepcopy(make_es)


@pytest.fixture
def pd_int_es(make_int_es):
    return copy.deepcopy(make_int_es)


@pytest.fixture
def dask_int_es(pd_int_es):
    es = EntitySet(id=pd_int_es.id)
    for df in pd_int_es.dataframes:
        dd_df = dd.from_pandas(df.reset_index(drop=True), npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(dd_df)

    for rel in pd_int_es.relationships:
        es.add_relationship(
            rel.parent_dataframe.ww.name,
            rel._parent_column_name,
            rel.child_dataframe.ww.name,
            rel._child_column_name,
        )
    return es


@pytest.fixture
def spark_int_es(pd_int_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    es = EntitySet(id=pd_int_es.id)
    for df in pd_int_es.dataframes:
        cleaned_df = pd_to_spark_clean(df).reset_index(drop=True)
        spark_df = ps.from_pandas(cleaned_df)
        spark_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(spark_df)

    for rel in pd_int_es.relationships:
        es.add_relationship(
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
    return es


@pytest.fixture(params=["pd_int_es", "dask_int_es", "spark_int_es"])
def int_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dask_es(pd_es):
    es = EntitySet(id=pd_es.id)
    for df in pd_es.dataframes:
        dd_df = dd.from_pandas(df.reset_index(drop=True), npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(dd_df)

    for rel in pd_es.relationships:
        es.add_relationship(
            rel.parent_dataframe.ww.name,
            rel._parent_column_name,
            rel.child_dataframe.ww.name,
            rel._child_column_name,
        )
    return es


@pytest.fixture
def spark_es(pd_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    es = EntitySet(id=pd_es.id)
    for df in pd_es.dataframes:
        cleaned_df = pd_to_spark_clean(df).reset_index(drop=True)
        spark_df = ps.from_pandas(cleaned_df)
        spark_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(spark_df)

    for rel in pd_es.relationships:
        es.add_relationship(
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
    return es


@pytest.fixture(params=["pd_es", "dask_es", "spark_es"])
def es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_latlong_df():
    df = pd.DataFrame({"idx": [0, 1, 2], "latLong": [pd.NA, (1, 2), (pd.NA, pd.NA)]})

    return df


@pytest.fixture
def dask_latlong_df(pd_latlong_df):
    return dd.from_pandas(pd_latlong_df.reset_index(drop=True), npartitions=4)


@pytest.fixture
def spark_latlong_df(pd_latlong_df):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    cleaned_df = pd_to_spark_clean(pd_latlong_df)

    pdf = ps.from_pandas(cleaned_df)

    return pdf


@pytest.fixture(params=["pd_latlong_df", "dask_latlong_df", "spark_latlong_df"])
def latlong_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["pd_diamond_es", "dask_diamond_es", "spark_diamond_es"])
def diamond_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_diamond_es():
    countries_df = pd.DataFrame({"id": range(2), "name": ["US", "Canada"]})
    regions_df = pd.DataFrame(
        {
            "id": range(3),
            "country_id": [0, 0, 1],
            "name": ["Northeast", "South", "Quebec"],
        },
    ).astype({"name": "category"})
    stores_df = pd.DataFrame(
        {
            "id": range(5),
            "region_id": [0, 1, 2, 2, 1],
            "square_ft": [2000, 3000, 1500, 2500, 2700],
        },
    )
    customers_df = pd.DataFrame(
        {
            "id": range(5),
            "region_id": [1, 0, 0, 1, 1],
            "name": ["A", "B", "C", "D", "E"],
        },
    )
    transactions_df = pd.DataFrame(
        {
            "id": range(8),
            "store_id": [4, 4, 2, 3, 4, 0, 1, 1],
            "customer_id": [3, 0, 2, 4, 3, 3, 2, 3],
            "amount": [100, 40, 45, 83, 13, 94, 27, 81],
        },
    )

    dataframes = {
        "countries": (countries_df, "id"),
        "regions": (regions_df, "id"),
        "stores": (stores_df, "id"),
        "customers": (customers_df, "id"),
        "transactions": (transactions_df, "id"),
    }
    relationships = [
        ("countries", "id", "regions", "country_id"),
        ("regions", "id", "stores", "region_id"),
        ("regions", "id", "customers", "region_id"),
        ("stores", "id", "transactions", "store_id"),
        ("customers", "id", "transactions", "customer_id"),
    ]
    return EntitySet(
        id="ecommerce_diamond",
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def dask_diamond_es(pd_diamond_es):
    dataframes = {}
    for df in pd_diamond_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=2)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_diamond_es.relationships
    ]

    return EntitySet(
        id=pd_diamond_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def spark_diamond_es(pd_diamond_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    dataframes = {}
    for df in pd_diamond_es.dataframes:
        spark_df = ps.from_pandas(pd_to_spark_clean(df))
        spark_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (spark_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_diamond_es.relationships
    ]

    return EntitySet(
        id=pd_diamond_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture(
    params=["pd_default_value_es", "dask_default_value_es", "spark_default_value_es"],
)
def default_value_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_default_value_es():
    transactions = pd.DataFrame(
        {"id": [1, 2, 3, 4], "session_id": ["a", "a", "b", "c"], "value": [1, 1, 1, 1]},
    )

    sessions = pd.DataFrame({"id": ["a", "b"]})

    es = EntitySet()
    es.add_dataframe(dataframe_name="transactions", dataframe=transactions, index="id")
    es.add_dataframe(dataframe_name="sessions", dataframe=sessions, index="id")

    es.add_relationship("sessions", "id", "transactions", "session_id")
    return es


@pytest.fixture
def dask_default_value_es(pd_default_value_es):
    dataframes = {}
    for df in pd_default_value_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_default_value_es.relationships
    ]

    return EntitySet(
        id=pd_default_value_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def spark_default_value_es(pd_default_value_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    dataframes = {}
    for df in pd_default_value_es.dataframes:
        spark_df = ps.from_pandas(pd_to_spark_clean(df))
        spark_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (spark_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_default_value_es.relationships
    ]

    return EntitySet(
        id=pd_default_value_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture(
    params=["pd_home_games_es", "dask_home_games_es", "spark_home_games_es"],
)
def home_games_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_home_games_es():
    teams = pd.DataFrame({"id": range(3), "name": ["Breakers", "Spirit", "Thorns"]})
    games = pd.DataFrame(
        {
            "id": range(5),
            "home_team_id": [2, 2, 1, 0, 1],
            "away_team_id": [1, 0, 2, 1, 0],
            "home_team_score": [3, 0, 1, 0, 4],
            "away_team_score": [2, 1, 2, 0, 0],
        },
    )
    dataframes = {"teams": (teams, "id"), "games": (games, "id")}
    relationships = [("teams", "id", "games", "home_team_id")]
    return EntitySet(dataframes=dataframes, relationships=relationships)


@pytest.fixture
def dask_home_games_es(pd_home_games_es):
    dataframes = {}
    for df in pd_home_games_es.dataframes:
        dd_df = dd.from_pandas(df, npartitions=2)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (dd_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_home_games_es.relationships
    ]

    return EntitySet(
        id=pd_home_games_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def spark_home_games_es(pd_home_games_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    dataframes = {}
    for df in pd_home_games_es.dataframes:
        spark_df = ps.from_pandas(pd_to_spark_clean(df))
        spark_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (spark_df,)

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_home_games_es.relationships
    ]

    return EntitySet(
        id=pd_home_games_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def games_es(home_games_es):
    return home_games_es.add_relationship("teams", "id", "games", "away_team_id")


@pytest.fixture
def pd_mock_customer():
    return demo.load_mock_customer(return_entityset=True, random_seed=0)


@pytest.fixture
def dd_mock_customer(pd_mock_customer):
    dataframes = {}
    for df in pd_mock_customer.dataframes:
        dd_df = dd.from_pandas(df.reset_index(drop=True), npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        dataframes[df.ww.name] = (
            dd_df,
            df.ww.index,
            df.ww.time_index,
            df.ww.logical_types,
        )
    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_mock_customer.relationships
    ]

    return EntitySet(
        id=pd_mock_customer.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def spark_mock_customer(pd_mock_customer):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    dataframes = {}
    for df in pd_mock_customer.dataframes:
        cleaned_df = pd_to_spark_clean(df).reset_index(drop=True)
        dataframes[df.ww.name] = (
            ps.from_pandas(cleaned_df),
            df.ww.index,
            df.ww.time_index,
            df.ww.logical_types,
        )

    relationships = [
        (
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
        for rel in pd_mock_customer.relationships
    ]

    return EntitySet(
        id=pd_mock_customer.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture(params=["pd_mock_customer", "dd_mock_customer", "spark_mock_customer"])
def mock_customer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def lt(es):
    def label_func(df):
        return df["value"].sum() > 10

    lm = cp.LabelMaker(
        target_dataframe_name="id",
        time_index="datetime",
        labeling_function=label_func,
        window_size="1m",
    )

    df = es["log"]
    df = to_pandas(df)
    labels = lm.search(df, num_examples_per_instance=-1)
    labels = labels.rename(columns={"cutoff_time": "time"})
    return labels


@pytest.fixture(params=["pd_dataframes", "dask_dataframes", "spark_dataframes"])
def dataframes(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_dataframes():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "fraud": [True, False, False, False, True, True],
        },
    )
    dataframes = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time"),
    }
    return dataframes


@pytest.fixture
def dask_dataframes():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "fraud": [True, False, False, False, True, True],
        },
    )
    cards_df = dd.from_pandas(cards_df, npartitions=2)
    transactions_df = dd.from_pandas(transactions_df, npartitions=2)

    cards_ltypes = {"id": Integer}
    transactions_ltypes = {
        "id": Integer,
        "card_id": Integer,
        "transaction_time": Integer,
        "fraud": Boolean,
    }

    dataframes = {
        "cards": (cards_df, "id", None, cards_ltypes),
        "transactions": (
            transactions_df,
            "id",
            "transaction_time",
            transactions_ltypes,
        ),
    }
    return dataframes


@pytest.fixture
def spark_dataframes():
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    cards_df = ps.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = ps.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "fraud": [True, False, False, False, True, True],
        },
    )
    cards_ltypes = {"id": Integer}
    transactions_ltypes = {
        "id": Integer,
        "card_id": Integer,
        "transaction_time": Integer,
        "fraud": Boolean,
    }

    dataframes = {
        "cards": (cards_df, "id", None, cards_ltypes),
        "transactions": (
            transactions_df,
            "id",
            "transaction_time",
            transactions_ltypes,
        ),
    }
    return dataframes


@pytest.fixture
def relationships():
    return [("cards", "id", "transactions", "card_id")]


@pytest.fixture(params=["pd_transform_es", "dask_transform_es", "spark_transform_es"])
def transform_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_transform_es():
    # Create dataframe
    df = pd.DataFrame(
        {
            "a": [14, 12, 10],
            "b": [False, False, True],
            "b1": [True, True, False],
            "b12": [4, 5, 6],
            "P": [10, 15, 12],
        },
    )
    es = EntitySet(id="test")
    # Add dataframe to entityset
    es.add_dataframe(
        dataframe_name="first",
        dataframe=df,
        index="index",
        make_index=True,
    )

    return es


@pytest.fixture
def dask_transform_es(pd_transform_es):
    es = EntitySet(id=pd_transform_es.id)
    for df in pd_transform_es.dataframes:
        es.add_dataframe(
            dataframe_name=df.ww.name,
            dataframe=dd.from_pandas(df, npartitions=2),
            index=df.ww.index,
            logical_types=df.ww.logical_types,
        )
    return es


@pytest.fixture
def spark_transform_es(pd_transform_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    es = EntitySet(id=pd_transform_es.id)
    for df in pd_transform_es.dataframes:
        es.add_dataframe(
            dataframe_name=df.ww.name,
            dataframe=ps.from_pandas(df),
            index=df.ww.index,
            logical_types=df.ww.logical_types,
        )
    return es


@pytest.fixture(
    params=[
        "divide_by_zero_es_pd",
        "divide_by_zero_es_dask",
        "divide_by_zero_es_spark",
    ],
)
def divide_by_zero_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def divide_by_zero_es_pd():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "col1": [1, 0, -3, 4],
            "col2": [0, 0, 0, 4],
        },
    )
    return EntitySet("data", {"zero": (df, "id", None)})


@pytest.fixture
def divide_by_zero_es_dask(divide_by_zero_es_pd):
    es = EntitySet(id=divide_by_zero_es_pd.id)
    for df in divide_by_zero_es_pd.dataframes:
        es.add_dataframe(
            dataframe_name=df.ww.name,
            dataframe=dd.from_pandas(df, npartitions=2),
            index=df.ww.index,
            logical_types=df.ww.logical_types,
        )
    return es


@pytest.fixture
def divide_by_zero_es_spark(divide_by_zero_es_pd):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    es = EntitySet(id=divide_by_zero_es_pd.id)
    for df in divide_by_zero_es_pd.dataframes:
        es.add_dataframe(
            dataframe_name=df.ww.name,
            dataframe=ps.from_pandas(df),
            index=df.ww.index,
            logical_types=df.ww.logical_types,
        )
    return es


@pytest.fixture
def rolling_series_pd():
    return pd.Series(
        range(20),
        index=pd.date_range(start="2020-01-01", end="2020-01-20"),
    )
