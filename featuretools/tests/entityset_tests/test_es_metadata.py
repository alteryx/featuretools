import pandas as pd
import pytest

from featuretools import EntitySet
from featuretools.tests.testing_utils import backward_path, forward_path


def test_cannot_re_add_relationships_that_already_exists(es):
    before_len = len(es.relationships)
    es.add_relationship(relationship=es.relationships[0])
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        assert r.parent_dataframe.ww.index == r._parent_column_name
        assert "foreign_key" in r.child_column.ww.semantic_tags
        assert r.child_column.ww.logical_type == r.parent_column.ww.logical_type


def test_get_forward_dataframes(es):
    dataframes = es.get_forward_dataframes("log")
    path_to_sessions = forward_path(es, ["log", "sessions"])
    path_to_products = forward_path(es, ["log", "products"])
    assert list(dataframes) == [
        ("sessions", path_to_sessions),
        ("products", path_to_products),
    ]


def test_get_backward_dataframes(es):
    dataframes = es.get_backward_dataframes("customers")
    path_to_sessions = backward_path(es, ["customers", "sessions"])
    assert list(dataframes) == [("sessions", path_to_sessions)]


def test_get_forward_dataframes_deep(es):
    dataframes = es.get_forward_dataframes("log", deep=True)
    path_to_sessions = forward_path(es, ["log", "sessions"])
    path_to_products = forward_path(es, ["log", "products"])
    path_to_customers = forward_path(es, ["log", "sessions", "customers"])
    path_to_regions = forward_path(es, ["log", "sessions", "customers", "régions"])
    path_to_cohorts = forward_path(es, ["log", "sessions", "customers", "cohorts"])
    assert list(dataframes) == [
        ("sessions", path_to_sessions),
        ("customers", path_to_customers),
        ("cohorts", path_to_cohorts),
        ("régions", path_to_regions),
        ("products", path_to_products),
    ]


def test_get_backward_dataframes_deep(es):
    dataframes = es.get_backward_dataframes("customers", deep=True)
    path_to_log = backward_path(es, ["customers", "sessions", "log"])
    path_to_sessions = backward_path(es, ["customers", "sessions"])
    assert list(dataframes) == [("sessions", path_to_sessions), ("log", path_to_log)]


def test_get_forward_relationships(es):
    relationships = es.get_forward_relationships("log")
    assert len(relationships) == 2
    assert relationships[0]._parent_dataframe_name == "sessions"
    assert relationships[0]._child_dataframe_name == "log"
    assert relationships[1]._parent_dataframe_name == "products"
    assert relationships[1]._child_dataframe_name == "log"

    relationships = es.get_forward_relationships("sessions")
    assert len(relationships) == 1
    assert relationships[0]._parent_dataframe_name == "customers"
    assert relationships[0]._child_dataframe_name == "sessions"


def test_get_backward_relationships(es):
    relationships = es.get_backward_relationships("sessions")
    assert len(relationships) == 1
    assert relationships[0]._parent_dataframe_name == "sessions"
    assert relationships[0]._child_dataframe_name == "log"

    relationships = es.get_backward_relationships("customers")
    assert len(relationships) == 1
    assert relationships[0]._parent_dataframe_name == "customers"
    assert relationships[0]._child_dataframe_name == "sessions"


def test_find_forward_paths(es):
    paths = list(es.find_forward_paths("log", "customers"))
    assert len(paths) == 1

    path = paths[0]

    assert len(path) == 2
    assert path[0]._child_dataframe_name == "log"
    assert path[0]._parent_dataframe_name == "sessions"
    assert path[1]._child_dataframe_name == "sessions"
    assert path[1]._parent_dataframe_name == "customers"


def test_find_forward_paths_multiple_paths(diamond_es):
    paths = list(diamond_es.find_forward_paths("transactions", "regions"))
    assert len(paths) == 2

    path1, path2 = paths

    r1, r2 = path1
    assert r1._child_dataframe_name == "transactions"
    assert r1._parent_dataframe_name == "stores"
    assert r2._child_dataframe_name == "stores"
    assert r2._parent_dataframe_name == "regions"

    r1, r2 = path2
    assert r1._child_dataframe_name == "transactions"
    assert r1._parent_dataframe_name == "customers"
    assert r2._child_dataframe_name == "customers"
    assert r2._parent_dataframe_name == "regions"


def test_find_forward_paths_multiple_relationships(games_es):
    paths = list(games_es.find_forward_paths("games", "teams"))
    assert len(paths) == 2

    path1, path2 = paths
    assert len(path1) == 1
    assert len(path2) == 1
    r1 = path1[0]
    r2 = path2[0]

    assert r1._child_dataframe_name == "games"
    assert r2._child_dataframe_name == "games"
    assert r1._parent_dataframe_name == "teams"
    assert r2._parent_dataframe_name == "teams"

    assert r1._child_column_name == "home_team_id"
    assert r2._child_column_name == "away_team_id"
    assert r1._parent_column_name == "id"
    assert r2._parent_column_name == "id"


@pytest.fixture
def pd_employee_df():
    return pd.DataFrame({"id": [0], "manager_id": [0]})


@pytest.fixture
def dd_employee_df(pd_employee_df):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_employee_df, npartitions=2)


@pytest.fixture
def spark_employee_df(pd_employee_df):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_employee_df)


@pytest.fixture(params=["pd_employee_df", "dd_employee_df", "spark_employee_df"])
def employee_df(request):
    return request.getfixturevalue(request.param)


def test_find_forward_paths_ignores_loops(employee_df):
    dataframes = {"employees": (employee_df, "id")}
    relationships = [("employees", "id", "employees", "manager_id")]
    es = EntitySet(dataframes=dataframes, relationships=relationships)

    paths = list(es.find_forward_paths("employees", "employees"))
    assert len(paths) == 1
    assert paths[0] == []


def test_find_backward_paths(es):
    paths = list(es.find_backward_paths("customers", "log"))
    assert len(paths) == 1

    path = paths[0]

    assert len(path) == 2
    assert path[0]._child_dataframe_name == "sessions"
    assert path[0]._parent_dataframe_name == "customers"
    assert path[1]._child_dataframe_name == "log"
    assert path[1]._parent_dataframe_name == "sessions"


def test_find_backward_paths_multiple_paths(diamond_es):
    paths = list(diamond_es.find_backward_paths("regions", "transactions"))
    assert len(paths) == 2

    path1, path2 = paths

    r1, r2 = path1
    assert r1._child_dataframe_name == "stores"
    assert r1._parent_dataframe_name == "regions"
    assert r2._child_dataframe_name == "transactions"
    assert r2._parent_dataframe_name == "stores"

    r1, r2 = path2
    assert r1._child_dataframe_name == "customers"
    assert r1._parent_dataframe_name == "regions"
    assert r2._child_dataframe_name == "transactions"
    assert r2._parent_dataframe_name == "customers"


def test_find_backward_paths_multiple_relationships(games_es):
    paths = list(games_es.find_backward_paths("teams", "games"))
    assert len(paths) == 2

    path1, path2 = paths
    assert len(path1) == 1
    assert len(path2) == 1
    r1 = path1[0]
    r2 = path2[0]

    assert r1._child_dataframe_name == "games"
    assert r2._child_dataframe_name == "games"
    assert r1._parent_dataframe_name == "teams"
    assert r2._parent_dataframe_name == "teams"

    assert r1._child_column_name == "home_team_id"
    assert r2._child_column_name == "away_team_id"
    assert r1._parent_column_name == "id"
    assert r2._parent_column_name == "id"


def test_has_unique_path(diamond_es):
    assert diamond_es.has_unique_forward_path("customers", "regions")
    assert not diamond_es.has_unique_forward_path("transactions", "regions")


def test_raise_key_error_missing_dataframe(es):
    error_text = "DataFrame testing does not exist in ecommerce"
    with pytest.raises(KeyError, match=error_text):
        es["testing"]

    es_without_id = EntitySet()
    error_text = "DataFrame testing does not exist in entity set"
    with pytest.raises(KeyError, match=error_text):
        es_without_id["testing"]


def test_add_parent_not_index_column(es):
    error_text = "Parent column 'language' is not the index of dataframe régions"
    with pytest.raises(AttributeError, match=error_text):
        es.add_relationship("régions", "language", "customers", "région_id")
