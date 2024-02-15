import contextlib
import copy
import os

import composeml as cp
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Integer

from featuretools import EntitySet, demo
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.tests.testing_utils import make_ecommerce_entityset, to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.spark_utils import pd_to_spark_clean


@pytest.fixture()
def dask_cluster():
    distributed = pytest.importorskip(
        "distributed",
        reason="Dask not installed, skipping",
    )
    if distributed:
        with distributed.LocalCluster() as cluster:
            yield cluster


@pytest.fixture()
def three_worker_dask_cluster():
    distributed = pytest.importorskip(
        "distributed",
        reason="Dask not installed, skipping",
    )
    if distributed:
        with distributed.LocalCluster(n_workers=3) as cluster:
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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

    kwargs = {
        "time_index": "datetime",
        "labeling_function": label_func,
        "window_size": "1m",
    }
    if parse(cp.__version__) >= parse("0.10.0"):
        kwargs["target_dataframe_index"] = "id"
    else:
        kwargs["target_dataframe_name"] = "id"  # pragma: no cover

    lm = cp.LabelMaker(**kwargs)

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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
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
def window_series_pd():
    return pd.Series(
        range(20),
        index=pd.date_range(start="2020-01-01", end="2020-01-20"),
    )


@pytest.fixture
def window_date_range_pd():
    return pd.date_range(start="2022-11-1", end="2022-11-5", periods=30)


@pytest.fixture
def rolling_outlier_series_pd():
    return pd.Series(
        [0] * 4 + [10] + [0] * 4 + [10] + [0] * 5,
        index=pd.date_range(start="2020-01-01", end="2020-01-15", periods=15),
    )


@pytest.fixture
def postal_code_dataframe_pd():
    df = pd.DataFrame(
        {
            "string_dtype": pd.Series(["90210", "60018", "10010", "92304-4201"]),
            "int_dtype": pd.Series([10000, 20000, 30000]).astype("category"),
            "has_nulls": pd.Series([np.nan, 20000, 30000]).astype("category"),
        },
    )
    return df


@pytest.fixture
def postal_code_dataframe_pyspark(postal_code_dataframe_pd):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    df = ps.from_pandas(postal_code_dataframe_pd)
    return df


@pytest.fixture
def postal_code_dataframe_dask(postal_code_dataframe_pd):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    df = dd.from_pandas(
        postal_code_dataframe_pd,
        npartitions=1,
    ).categorize()
    return df


@pytest.fixture(
    params=[
        "postal_code_dataframe_pd",
        "postal_code_dataframe_pyspark",
        "postal_code_dataframe_dask",
    ],
)
def postal_code_dataframe(request):
    df = request.getfixturevalue(request.param)
    df.ww.init(
        logical_types={
            "string_dtype": "PostalCode",
            "int_dtype": "PostalCode",
            "has_nulls": "PostalCode",
        },
    )
    return df


def create_test_credentials(test_path):
    with open(test_path, "w+") as f:
        f.write("[test]\n")
        f.write("aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n")
        f.write("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")


def create_test_config(test_path_config):
    with open(test_path_config, "w+") as f:
        f.write("[profile test]\n")
        f.write("region=us-east-2\n")
        f.write("output=text\n")


@pytest.fixture
def setup_test_profile(monkeypatch, tmp_path):
    cache = tmp_path.joinpath(".cache")
    cache.mkdir()
    test_path = str(cache.joinpath("test_credentials"))
    test_path_config = str(cache.joinpath("test_config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", test_path)
    monkeypatch.setenv("AWS_CONFIG_FILE", test_path_config)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "test")

    with contextlib.suppress(OSError):
        os.remove(test_path)
        os.remove(test_path_config)  # pragma: no cover

    create_test_credentials(test_path)
    create_test_config(test_path_config)
    yield
    os.remove(test_path)
    os.remove(test_path_config)


@pytest.fixture
def test_aggregation_primitive():
    class TestAgg(AggregationPrimitive):
        name = "test"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        stack_on = []

    return TestAgg


@pytest.fixture
def test_transform_primitive():
    class TestTransform(TransformPrimitive):
        name = "test"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        stack_on = []

    return TestTransform


@pytest.fixture
def strings_that_have_triggered_errors_before():
    return [
        "    ",
        '"This Borderlands game here"" is the perfect conclusion to the ""Borderlands 3"" line, which focuses on the fans ""favorite character and gives the players the opportunity to close for a long time some very important questions about\'s character and the memorable scenery with which the players interact.',
    ]
