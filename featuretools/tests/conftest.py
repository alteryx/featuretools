import contextlib
import copy
import os

import composeml as cp
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse
from woodwork.column_schema import ColumnSchema

from featuretools import EntitySet, demo
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.tests.testing_utils import make_ecommerce_entityset


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


@pytest.fixture(scope="session")
def make_es():
    return make_ecommerce_entityset()


@pytest.fixture(scope="session")
def make_int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


@pytest.fixture
def es(make_es):
    return copy.deepcopy(make_es)


@pytest.fixture
def int_es(make_int_es):
    return copy.deepcopy(make_int_es)


@pytest.fixture
def latlong_df():
    df = pd.DataFrame({"idx": [0, 1, 2], "latLong": [pd.NA, (1, 2), (pd.NA, pd.NA)]})
    return df


@pytest.fixture
def diamond_es():
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
def default_value_es():
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
def home_games_es():
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
def games_es(home_games_es):
    return home_games_es.add_relationship("teams", "id", "games", "away_team_id")


@pytest.fixture
def mock_customer():
    return demo.load_mock_customer(return_entityset=True, random_seed=0)


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
    labels = lm.search(df, num_examples_per_instance=-1)
    labels = labels.rename(columns={"cutoff_time": "time"})
    return labels


@pytest.fixture
def dataframes():
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
def relationships():
    return [("cards", "id", "transactions", "card_id")]


@pytest.fixture
def transform_es():
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
def divide_by_zero_es():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "col1": [1, 0, -3, 4],
            "col2": [0, 0, 0, 4],
        },
    )
    return EntitySet("data", {"zero": (df, "id", None)})


@pytest.fixture
def window_series():
    return pd.Series(
        range(20),
        index=pd.date_range(start="2020-01-01", end="2020-01-20"),
    )


@pytest.fixture
def window_date_range():
    return pd.date_range(start="2022-11-1", end="2022-11-5", periods=30)


@pytest.fixture
def rolling_outlier_series():
    return pd.Series(
        [0] * 4 + [10] + [0] * 4 + [10] + [0] * 5,
        index=pd.date_range(start="2020-01-01", end="2020-01-15", periods=15),
    )


@pytest.fixture
def postal_code_dataframe():
    df = pd.DataFrame(
        {
            "string_dtype": pd.Series(["90210", "60018", "10010", "92304-4201"]),
            "int_dtype": pd.Series([10000, 20000, 30000]).astype("category"),
            "has_nulls": pd.Series([np.nan, 20000, 30000]).astype("category"),
        },
    )
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
