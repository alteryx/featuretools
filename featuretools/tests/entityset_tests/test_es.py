import copy
import logging
import pickle
import re
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PostalCode,
    SubRegionCode,
)

from featuretools import Relationship
from featuretools.demo import load_retail
from featuretools.entityset import EntitySet
from featuretools.entityset.entityset import LTI_COLUMN_NAME, WW_SCHEMA_KEY
from featuretools.tests.testing_utils import get_df_tags, to_pandas
from featuretools.utils.gen_utils import Library, import_or_none, is_instance
from featuretools.utils.spark_utils import pd_to_spark_clean

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def test_normalize_time_index_as_additional_column(es):
    error_text = "Not moving signup_date as it is the base time index column. Perhaps, move the column to the copy_columns."
    with pytest.raises(ValueError, match=error_text):
        assert "signup_date" in es["customers"].columns
        es.normalize_dataframe(
            base_dataframe_name="customers",
            new_dataframe_name="cancellations",
            index="cancel_reason",
            make_time_index="signup_date",
            additional_columns=["signup_date"],
            copy_columns=[],
        )


def test_normalize_time_index_as_copy_column(es):
    assert "signup_date" in es["customers"].columns
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancellations",
        index="cancel_reason",
        make_time_index="signup_date",
        additional_columns=[],
        copy_columns=["signup_date"],
    )

    assert "signup_date" in es["customers"].columns
    assert es["customers"].ww.time_index == "signup_date"
    assert "signup_date" in es["cancellations"].columns
    assert es["cancellations"].ww.time_index == "signup_date"


def test_normalize_time_index_as_copy_column_new_time_index(es):
    assert "signup_date" in es["customers"].columns
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancellations",
        index="cancel_reason",
        make_time_index=True,
        additional_columns=[],
        copy_columns=["signup_date"],
    )

    assert "signup_date" in es["customers"].columns
    assert es["customers"].ww.time_index == "signup_date"
    assert "first_customers_time" in es["cancellations"].columns
    assert "signup_date" not in es["cancellations"].columns
    assert es["cancellations"].ww.time_index == "first_customers_time"


def test_normalize_time_index_as_copy_column_no_time_index(es):
    assert "signup_date" in es["customers"].columns
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancellations",
        index="cancel_reason",
        make_time_index=False,
        additional_columns=[],
        copy_columns=["signup_date"],
    )

    assert "signup_date" in es["customers"].columns
    assert es["customers"].ww.time_index == "signup_date"
    assert "signup_date" in es["cancellations"].columns
    assert es["cancellations"].ww.time_index is None


def test_cannot_re_add_relationships_that_already_exists(es):
    warn_text = "Not adding duplicate relationship: " + str(es.relationships[0])
    before_len = len(es.relationships)
    rel = es.relationships[0]
    with pytest.warns(UserWarning, match=warn_text):
        es.add_relationship(relationship=rel)
    with pytest.warns(UserWarning, match=warn_text):
        es.add_relationship(
            rel._parent_dataframe_name,
            rel._parent_column_name,
            rel._child_dataframe_name,
            rel._child_column_name,
        )
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        parent_df = es[r.parent_dataframe.ww.name]
        child_df = es[r.child_dataframe.ww.name]
        assert parent_df.ww.index == r._parent_column_name
        assert "foreign_key" in r.child_column.ww.semantic_tags
        assert str(parent_df[r._parent_column_name].dtype) == str(
            child_df[r._child_column_name].dtype,
        )


def test_add_relationship_diff_param_logical_types(es):
    ordinal_1 = Ordinal(order=[0, 1, 2, 3, 4, 5, 6])
    ordinal_2 = Ordinal(order=[0, 1, 2, 3, 4, 5])
    es["sessions"].ww.set_types(logical_types={"id": ordinal_1})
    log_2_df = es["log"].copy()
    log_logical_types = {
        "id": Integer,
        "session_id": ordinal_2,
        "product_id": Categorical(),
        "datetime": Datetime,
        "value": Double,
        "value_2": Double,
        "latlong": LatLong,
        "latlong2": LatLong,
        "zipcode": PostalCode,
        "countrycode": CountryCode,
        "subregioncode": SubRegionCode,
        "value_many_nans": Double,
        "priority_level": Ordinal(order=[0, 1, 2]),
        "purchased": Boolean,
        "comments": NaturalLanguage,
        "url": URL,
        "email_address": EmailAddress,
    }
    log_semantic_tags = {"session_id": "foreign_key", "product_id": "foreign_key"}
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(
        dataframe_name="log2",
        dataframe=log_2_df,
        index="id",
        logical_types=log_logical_types,
        semantic_tags=log_semantic_tags,
        time_index="datetime",
    )
    assert "log2" in es.dataframe_dict
    assert es["log2"].ww.schema is not None
    assert isinstance(es["log2"].ww.logical_types["session_id"], Ordinal)
    assert isinstance(es["sessions"].ww.logical_types["id"], Ordinal)
    assert (
        es["sessions"].ww.logical_types["id"]
        != es["log2"].ww.logical_types["session_id"]
    )

    warning_text = "Changing child logical type to match parent."
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship("sessions", "id", "log2", "session_id")
    assert isinstance(es["log2"].ww.logical_types["product_id"], Categorical)
    assert isinstance(es["products"].ww.logical_types["id"], Categorical)


def test_add_relationship_different_logical_types_same_dtype(es):
    log_2_df = es["log"].copy()
    log_logical_types = {
        "id": Integer,
        "session_id": Integer,
        "product_id": CountryCode,
        "datetime": Datetime,
        "value": Double,
        "value_2": Double,
        "latlong": LatLong,
        "latlong2": LatLong,
        "zipcode": PostalCode,
        "countrycode": CountryCode,
        "subregioncode": SubRegionCode,
        "value_many_nans": Double,
        "priority_level": Ordinal(order=[0, 1, 2]),
        "purchased": Boolean,
        "comments": NaturalLanguage,
        "url": URL,
        "email_address": EmailAddress,
    }
    log_semantic_tags = {"session_id": "foreign_key", "product_id": "foreign_key"}
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(
        dataframe_name="log2",
        dataframe=log_2_df,
        index="id",
        logical_types=log_logical_types,
        semantic_tags=log_semantic_tags,
        time_index="datetime",
    )
    assert "log2" in es.dataframe_dict
    assert es["log2"].ww.schema is not None
    assert isinstance(es["log2"].ww.logical_types["product_id"], CountryCode)
    assert isinstance(es["products"].ww.logical_types["id"], Categorical)

    warning_text = "Logical type CountryCode for child column product_id does not match parent column id logical type Categorical. Changing child logical type to match parent."
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship("products", "id", "log2", "product_id")
    assert isinstance(es["log2"].ww.logical_types["product_id"], Categorical)
    assert isinstance(es["products"].ww.logical_types["id"], Categorical)
    assert "foreign_key" in es["log2"].ww.semantic_tags["product_id"]


def test_add_relationship_different_compatible_dtypes(es):
    log_2_df = es["log"].copy()
    log_logical_types = {
        "id": Integer,
        "session_id": Datetime,
        "product_id": Categorical,
        "datetime": Datetime,
        "value": Double,
        "value_2": Double,
        "latlong": LatLong,
        "latlong2": LatLong,
        "zipcode": PostalCode,
        "countrycode": CountryCode,
        "subregioncode": SubRegionCode,
        "value_many_nans": Double,
        "priority_level": Ordinal(order=[0, 1, 2]),
        "purchased": Boolean,
        "comments": NaturalLanguage,
        "url": URL,
        "email_address": EmailAddress,
    }
    log_semantic_tags = {"session_id": "foreign_key", "product_id": "foreign_key"}
    assert set(log_logical_types) == set(log_2_df.columns)
    es.add_dataframe(
        dataframe_name="log2",
        dataframe=log_2_df,
        index="id",
        logical_types=log_logical_types,
        semantic_tags=log_semantic_tags,
        time_index="datetime",
    )
    assert "log2" in es.dataframe_dict
    assert es["log2"].ww.schema is not None
    assert isinstance(es["log2"].ww.logical_types["session_id"], Datetime)
    assert isinstance(es["customers"].ww.logical_types["id"], Integer)

    warning_text = "Logical type Datetime for child column session_id does not match parent column id logical type Integer. Changing child logical type to match parent."
    with pytest.warns(UserWarning, match=warning_text):
        es.add_relationship("customers", "id", "log2", "session_id")
    assert isinstance(es["log2"].ww.logical_types["session_id"], Integer)
    assert isinstance(es["customers"].ww.logical_types["id"], Integer)


def test_add_relationship_errors_child_v_index(es):
    new_df = es["log"].ww.copy()
    new_df.ww._schema.name = "log2"
    es.add_dataframe(dataframe=new_df)

    to_match = "Unable to add relationship because child column 'id' in 'log2' is also its index"
    with pytest.raises(ValueError, match=to_match):
        es.add_relationship("log", "id", "log2", "id")


def test_add_relationship_empty_child_convert_dtype(es):
    relationship = Relationship(es, "sessions", "id", "log", "session_id")
    empty_log_df = pd.DataFrame(columns=es["log"].columns)
    if es.dataframe_type == Library.DASK:
        empty_log_df = dd.from_pandas(empty_log_df, npartitions=2)
    elif es.dataframe_type == Library.SPARK:
        empty_log_df = ps.from_pandas(empty_log_df)

    es.add_dataframe(empty_log_df, "log")

    assert len(es["log"]) == 0
    # session_id will be Unknown logical type with dtype string
    assert es["log"]["session_id"].dtype == "string"

    es.relationships.remove(relationship)
    assert relationship not in es.relationships

    es.add_relationship(relationship=relationship)
    assert es["log"]["session_id"].dtype == "int64"


def test_add_relationship_with_relationship_object(es):
    relationship = Relationship(es, "sessions", "id", "log", "session_id")
    es.add_relationship(relationship=relationship)
    assert relationship in es.relationships


def test_add_relationships_with_relationship_object(es):
    relationships = [Relationship(es, "sessions", "id", "log", "session_id")]
    es.add_relationships(relationships)
    assert relationships[0] in es.relationships


def test_add_relationship_error(es):
    relationship = Relationship(es, "sessions", "id", "log", "session_id")
    error_message = (
        "Cannot specify dataframe and column name values and also supply a Relationship"
    )
    with pytest.raises(ValueError, match=error_message):
        es.add_relationship(parent_dataframe_name="sessions", relationship=relationship)


def test_query_by_values_returns_rows_in_given_order():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": ["a", "c", "b", "a", "a"],
            "time": [1000, 2000, 3000, 4000, 5000],
        },
    )

    es = EntitySet()
    es = es.add_dataframe(
        dataframe=data,
        dataframe_name="test",
        index="id",
        time_index="time",
        logical_types={"value": "Categorical"},
    )
    query = es.query_by_values("test", ["b", "a"], column_name="value")
    assert np.array_equal(query["id"], [1, 3, 4, 5])


def test_query_by_values_secondary_time_index(es):
    end = np.datetime64(datetime(2011, 10, 1))
    all_instances = [0, 1, 2]
    result = es.query_by_values("customers", all_instances, time_last=end)
    result = to_pandas(result, index="id")

    for col in ["cancel_date", "cancel_reason"]:
        nulls = result.loc[all_instances][col].isnull() == [False, True, True]
        assert nulls.all(), "Some instance has data it shouldn't for column %s" % col


def test_query_by_id(es):
    df = to_pandas(es.query_by_values("log", instance_vals=[0]))
    assert df["id"].values[0] == 0


def test_query_by_single_value(es):
    df = to_pandas(es.query_by_values("log", instance_vals=0))
    assert df["id"].values[0] == 0


def test_query_by_df(es):
    instance_df = pd.DataFrame({"id": [1, 3], "vals": [0, 1]})
    df = to_pandas(es.query_by_values("log", instance_vals=instance_df))

    assert np.array_equal(df["id"], [1, 3])


def test_query_by_id_with_time(es):
    df = es.query_by_values(
        dataframe_name="log",
        instance_vals=[0, 1, 2, 3, 4],
        time_last=datetime(2011, 4, 9, 10, 30, 2 * 6),
    )
    df = to_pandas(df)
    if es.dataframe_type == Library.SPARK:
        # Spark doesn't maintain order
        df = df.sort_values("id")

    assert list(df["id"].values) == [0, 1, 2]


def test_query_by_column_with_time(es):
    df = es.query_by_values(
        dataframe_name="log",
        instance_vals=[0, 1, 2],
        column_name="session_id",
        time_last=datetime(2011, 4, 9, 10, 50, 0),
    )
    df = to_pandas(df)

    true_values = [i * 5 for i in range(5)] + [i * 1 for i in range(4)] + [0]
    if es.dataframe_type == Library.SPARK:
        # Spark doesn't maintain order
        df = df.sort_values("id")

    assert list(df["id"].values) == list(range(10))
    assert list(df["value"].values) == true_values


def test_query_by_column_with_no_lti_and_training_window(es):
    match = (
        "Using training_window but last_time_index is not set for dataframe customers"
    )
    with pytest.warns(UserWarning, match=match):
        df = es.query_by_values(
            dataframe_name="customers",
            instance_vals=[0, 1, 2],
            column_name="cohort",
            time_last=datetime(2011, 4, 11),
            training_window="3d",
        )
    df = to_pandas(df)

    assert list(df["id"].values) == [1]
    assert list(df["age"].values) == [25]


def test_query_by_column_with_lti_and_training_window(es):
    es.add_last_time_indexes()
    df = es.query_by_values(
        dataframe_name="customers",
        instance_vals=[0, 1, 2],
        column_name="cohort",
        time_last=datetime(2011, 4, 11),
        training_window="3d",
    )
    # Account for different ordering between pandas and dask/spark
    df = to_pandas(df).reset_index(drop=True).sort_values("id")
    assert list(df["id"].values) == [0, 1, 2]
    assert list(df["age"].values) == [33, 25, 56]


def test_query_by_indexed_column(es):
    df = es.query_by_values(
        dataframe_name="log",
        instance_vals=["taco clock"],
        column_name="product_id",
    )
    # Account for different ordering between pandas and dask/spark
    df = to_pandas(df).reset_index(drop=True).sort_values("id")
    assert list(df["id"].values) == [15, 16]


@pytest.fixture
def pd_df():
    return pd.DataFrame({"id": [0, 1, 2], "category": ["a", "b", "c"]})


@pytest.fixture
def dd_df(pd_df):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_df, npartitions=2)


@pytest.fixture
def spark_df(pd_df):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_df)


@pytest.fixture(params=["pd_df", "dd_df", "spark_df"])
def df(request):
    return request.getfixturevalue(request.param)


def test_check_columns_and_dataframe(df):
    # matches
    logical_types = {"id": Integer, "category": Categorical}
    es = EntitySet(id="test")
    es.add_dataframe(
        df,
        dataframe_name="test_dataframe",
        index="id",
        logical_types=logical_types,
    )
    assert isinstance(
        es.dataframe_dict["test_dataframe"].ww.logical_types["category"],
        Categorical,
    )
    assert es.dataframe_dict["test_dataframe"].ww.semantic_tags["category"] == {
        "category",
    }


def test_make_index_any_location(df):
    logical_types = {"id": Integer, "category": Categorical}

    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="id1",
        make_index=True,
        logical_types=logical_types,
        dataframe=df,
    )
    if es.dataframe_type != Library.PANDAS:
        assert es.dataframe_dict["test_dataframe"].columns[-1] == "id1"
    else:
        assert es.dataframe_dict["test_dataframe"].columns[0] == "id1"

    assert es.dataframe_dict["test_dataframe"].ww.index == "id1"


def test_replace_dataframe_and_create_index(es):
    df = pd.DataFrame({"ints": [3, 4, 5], "category": ["a", "b", "a"]})
    final_df = df.copy()
    final_df["id"] = [0, 1, 2]
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=2)
    elif es.dataframe_type == Library.SPARK:
        df = ps.from_pandas(df)

    needs_idx_df = df.copy()

    logical_types = {"ints": Integer, "category": Categorical}
    es.add_dataframe(
        dataframe=df,
        dataframe_name="test_df",
        index="id",
        make_index=True,
        logical_types=logical_types,
    )

    assert es["test_df"].ww.index == "id"

    # DataFrame that needs the index column added
    assert "id" not in needs_idx_df.columns
    es.replace_dataframe("test_df", needs_idx_df)

    assert es["test_df"].ww.index == "id"
    df = to_pandas(es["test_df"]).sort_values(by="id")
    assert all(df["id"] == final_df["id"])
    assert all(df["ints"] == final_df["ints"])


def test_replace_dataframe_created_index_present(es):
    df = pd.DataFrame({"ints": [3, 4, 5], "category": ["a", "b", "a"]})
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=2)
    elif es.dataframe_type == Library.SPARK:
        df = ps.from_pandas(df)

    logical_types = {"ints": Integer, "category": Categorical}
    es.add_dataframe(
        dataframe=df,
        dataframe_name="test_df",
        index="id",
        make_index=True,
        logical_types=logical_types,
    )

    # DataFrame that already has the index column
    has_idx_df = es["test_df"].replace({0: 100})
    if es.dataframe_type == Library.PANDAS:
        has_idx_df.set_index("id", drop=False, inplace=True)

    assert "id" in has_idx_df.columns

    es.replace_dataframe("test_df", has_idx_df)
    assert es["test_df"].ww.index == "id"
    df = to_pandas(es["test_df"]).sort_values(by="ints")
    assert all(df["id"] == [100, 1, 2])


def test_index_any_location(df):
    logical_types = {"id": Integer, "category": Categorical}

    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="category",
        logical_types=logical_types,
        dataframe=df,
    )
    assert es.dataframe_dict["test_dataframe"].columns[1] == "category"
    assert es.dataframe_dict["test_dataframe"].ww.index == "category"


def test_extra_column_type(df):
    # more columns
    logical_types = {"id": Integer, "category": Categorical, "category2": Categorical}

    error_text = re.escape(
        "logical_types contains columns that are not present in dataframe: ['category2']",
    )
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id="test")
        es.add_dataframe(
            dataframe_name="test_dataframe",
            index="id",
            logical_types=logical_types,
            dataframe=df,
        )


def test_add_parent_not_index_column(es):
    error_text = "Parent column 'language' is not the index of dataframe régions"
    with pytest.raises(AttributeError, match=error_text):
        es.add_relationship("régions", "language", "customers", "région_id")


@pytest.fixture
def pd_df2():
    return pd.DataFrame({"category": [1, 2, 3], "category2": ["1", "2", "3"]})


@pytest.fixture
def dd_df2(pd_df2):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_df2, npartitions=2)


@pytest.fixture
def spark_df2(pd_df2):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_df2)


@pytest.fixture(params=["pd_df2", "dd_df2", "spark_df2"])
def df2(request):
    return request.getfixturevalue(request.param)


def test_none_index(df2):
    es = EntitySet(id="test")

    copy_df = df2.copy()
    copy_df.ww.init(name="test_dataframe")
    error_msg = "Cannot add Woodwork DataFrame to EntitySet without index"
    with pytest.raises(ValueError, match=error_msg):
        es.add_dataframe(dataframe=copy_df)

    warn_text = (
        "Using first column as index. To change this, specify the index parameter"
    )
    with pytest.warns(UserWarning, match=warn_text):
        es.add_dataframe(
            dataframe_name="test_dataframe",
            logical_types={"category": "Categorical"},
            dataframe=df2,
        )
    assert es["test_dataframe"].ww.index == "category"
    assert es["test_dataframe"].ww.semantic_tags["category"] == {"index"}
    assert isinstance(es["test_dataframe"].ww.logical_types["category"], Categorical)


@pytest.fixture
def pd_df3():
    return pd.DataFrame({"category": [1, 2, 3]})


@pytest.fixture
def dd_df3(pd_df3):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_df3, npartitions=2)


@pytest.fixture
def spark_df3(pd_df3):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_df3)


@pytest.fixture(params=["pd_df3", "dd_df3", "spark_df3"])
def df3(request):
    return request.getfixturevalue(request.param)


def test_unknown_index(df3):
    warn_text = "index id not found in dataframe, creating new integer column"
    es = EntitySet(id="test")
    with pytest.warns(UserWarning, match=warn_text):
        es.add_dataframe(
            dataframe_name="test_dataframe",
            dataframe=df3,
            index="id",
            logical_types={"category": "Categorical"},
        )
    assert es["test_dataframe"].ww.index == "id"
    assert list(to_pandas(es["test_dataframe"]["id"], sort_index=True)) == list(
        range(3),
    )


def test_doesnt_remake_index(df):
    logical_types = {"id": "Integer", "category": "Categorical"}
    error_text = "Cannot make index: column with name id already present"
    with pytest.raises(RuntimeError, match=error_text):
        es = EntitySet(id="test")
        es.add_dataframe(
            dataframe_name="test_dataframe",
            index="id",
            make_index=True,
            dataframe=df,
            logical_types=logical_types,
        )


def test_bad_time_index_column(df3):
    logical_types = {"category": "Categorical"}
    error_text = "Specified time index column `time` not found in dataframe"
    with pytest.raises(LookupError, match=error_text):
        es = EntitySet(id="test")
        es.add_dataframe(
            dataframe_name="test_dataframe",
            dataframe=df3,
            index="category",
            time_index="time",
            logical_types=logical_types,
        )


@pytest.fixture
def pd_df4():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "category": ["a", "b", "a"],
            "category_int": [1, 2, 3],
            "ints": ["1", "2", "3"],
            "floats": ["1", "2", "3.0"],
        },
    )
    df["category_int"] = df["category_int"].astype("category")
    return df


@pytest.fixture
def dd_df4(pd_df4):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_df4, npartitions=2)


@pytest.fixture
def spark_df4(pd_df4):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_to_spark_clean(pd_df4))


@pytest.fixture(params=["pd_df4", "dd_df4", "spark_df4"])
def df4(request):
    return request.getfixturevalue(request.param)


def test_converts_dtype_on_init(df4):
    logical_types = {"id": Integer, "ints": Integer, "floats": Double}
    if not isinstance(df4, pd.DataFrame):
        logical_types["category"] = Categorical
        logical_types["category_int"] = Categorical
    es = EntitySet(id="test")
    df4.ww.init(name="test_dataframe", index="id", logical_types=logical_types)
    es.add_dataframe(dataframe=df4)

    df = es["test_dataframe"]
    assert df["ints"].dtype.name == "int64"
    assert df["floats"].dtype.name == "float64"

    # this is infer from pandas dtype
    df = es["test_dataframe"]
    assert isinstance(df.ww.logical_types["category_int"], Categorical)


def test_converts_dtype_after_init(df4):
    category_dtype = "category"
    if ps and isinstance(df4, ps.DataFrame):
        category_dtype = "string"

    df4["category"] = df4["category"].astype(category_dtype)
    if not isinstance(df4, pd.DataFrame):
        logical_types = {
            "id": Integer,
            "category": Categorical,
            "category_int": Categorical,
            "ints": Integer,
            "floats": Double,
        }
    else:
        logical_types = None
    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="id",
        dataframe=df4,
        logical_types=logical_types,
    )
    df = es["test_dataframe"]

    df.ww.set_types(logical_types={"ints": "Integer"})
    assert isinstance(df.ww.logical_types["ints"], Integer)
    assert df["ints"].dtype == "int64"

    df.ww.set_types(logical_types={"ints": "Categorical"})
    assert isinstance(df.ww.logical_types["ints"], Categorical)
    assert df["ints"].dtype == category_dtype

    df.ww.set_types(logical_types={"ints": Ordinal(order=[1, 2, 3])})
    assert df.ww.logical_types["ints"] == Ordinal(order=[1, 2, 3])
    assert df["ints"].dtype == category_dtype

    df.ww.set_types(logical_types={"ints": "NaturalLanguage"})
    assert isinstance(df.ww.logical_types["ints"], NaturalLanguage)
    assert df["ints"].dtype == "string"


def test_warns_no_typing(df4):
    es = EntitySet(id="test")
    if not isinstance(df4, pd.DataFrame):
        msg = "Performing type inference on Dask or Spark DataFrames may be computationally intensive. Specify logical types for each column to speed up EntitySet initialization."
        with pytest.warns(UserWarning, match=msg):
            es.add_dataframe(dataframe_name="test_dataframe", index="id", dataframe=df4)
    else:
        es.add_dataframe(dataframe_name="test_dataframe", index="id", dataframe=df4)

    assert "test_dataframe" in es.dataframe_dict


@pytest.fixture
def pd_datetime1():
    times = pd.date_range("1/1/2011", periods=3, freq="H")
    time_strs = times.strftime("%Y-%m-%d")
    return pd.DataFrame({"id": [0, 1, 2], "time": time_strs})


@pytest.fixture
def dd_datetime1(pd_datetime1):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_datetime1, npartitions=2)


@pytest.fixture
def spark_datetime1(pd_datetime1):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_datetime1)


@pytest.fixture(params=["pd_datetime1", "dd_datetime1", "spark_datetime1"])
def datetime1(request):
    return request.getfixturevalue(request.param)


def test_converts_datetime(datetime1):
    # string converts to datetime correctly
    # This test fails without defining logical types.
    # Entityset infers time column should be numeric type
    logical_types = {"id": Integer, "time": Datetime}

    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="id",
        time_index="time",
        logical_types=logical_types,
        dataframe=datetime1,
    )
    pd_col = to_pandas(es["test_dataframe"]["time"])
    assert isinstance(es["test_dataframe"].ww.logical_types["time"], Datetime)
    assert type(pd_col[0]) == pd.Timestamp


@pytest.fixture
def pd_datetime2():
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp("Jan 2, 2011")
    time_strs = [actual.strftime(datetime_format)] * 3
    return pd.DataFrame(
        {"id": [0, 1, 2], "time_format": time_strs, "time_no_format": time_strs},
    )


@pytest.fixture
def dd_datetime2(pd_datetime2):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_datetime2, npartitions=2)


@pytest.fixture
def spark_datetime2(pd_datetime2):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_datetime2)


@pytest.fixture(params=["pd_datetime2", "dd_datetime2", "spark_datetime2"])
def datetime2(request):
    return request.getfixturevalue(request.param)


def test_handles_datetime_format(datetime2):
    # check if we load according to the format string
    # pass in an ambiguous date
    datetime_format = "%d-%m-%Y"
    actual = pd.Timestamp("Jan 2, 2011")

    logical_types = {
        "id": Integer,
        "time_format": (Datetime(datetime_format=datetime_format)),
        "time_no_format": Datetime,
    }

    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="id",
        logical_types=logical_types,
        dataframe=datetime2,
    )

    col_format = to_pandas(es["test_dataframe"]["time_format"])
    col_no_format = to_pandas(es["test_dataframe"]["time_no_format"])
    # without formatting pandas gets it wrong
    assert (col_no_format != actual).all()

    # with formatting we correctly get jan2
    assert (col_format == actual).all()


def test_handles_datetime_mismatch():
    # can't convert arbitrary strings
    df = pd.DataFrame({"id": [0, 1, 2], "time": ["a", "b", "tomorrow"]})
    logical_types = {"id": Integer, "time": Datetime}

    error_text = "Time index column must contain datetime or numeric values"
    with pytest.raises(TypeError, match=error_text):
        es = EntitySet(id="test")
        es.add_dataframe(
            df,
            dataframe_name="test_dataframe",
            index="id",
            time_index="time",
            logical_types=logical_types,
        )


def test_dataframe_init(es):
    df = pd.DataFrame(
        {
            "id": ["0", "1", "2"],
            "time": [datetime(2011, 4, 9, 10, 31, 3 * i) for i in range(3)],
            "category": ["a", "b", "a"],
            "number": [4, 5, 6],
        },
    )
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=2)
    elif es.dataframe_type == Library.SPARK:
        df = ps.from_pandas(df)
    logical_types = {"id": Categorical, "time": Datetime}
    if not isinstance(df, pd.DataFrame):
        extra_logical_types = {
            "category": Categorical,
            "number": Integer,
        }
        logical_types.update(extra_logical_types)
    es.add_dataframe(
        df.copy(),
        dataframe_name="test_dataframe",
        index="id",
        time_index="time",
        logical_types=logical_types,
    )
    if is_instance(df, dd, "DataFrame"):
        df_shape = (df.shape[0].compute(), df.shape[1])
    else:
        df_shape = df.shape
    if es.dataframe_type == Library.DASK:
        es_df_shape = (
            es["test_dataframe"].shape[0].compute(),
            es["test_dataframe"].shape[1],
        )
    else:
        es_df_shape = es["test_dataframe"].shape
    assert es_df_shape == df_shape
    assert es["test_dataframe"].ww.index == "id"
    assert es["test_dataframe"].ww.time_index == "time"
    assert set([v for v in es["test_dataframe"].ww.columns]) == set(df.columns)

    assert es["test_dataframe"]["time"].dtype == df["time"].dtype
    if es.dataframe_type == Library.SPARK:
        assert set(es["test_dataframe"]["id"].to_list()) == set(df["id"].to_list())
    else:
        assert set(es["test_dataframe"]["id"]) == set(df["id"])


@pytest.fixture
def pd_bad_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], 3: ["a", "b", "c"]})


@pytest.fixture
def dd_bad_df(pd_bad_df):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_bad_df, npartitions=2)


@pytest.fixture(params=["pd_bad_df", "dd_bad_df"])
def bad_df(request):
    return request.getfixturevalue(request.param)


# Skip for Spark, automatically converts non-str column names to str
def test_nonstr_column_names(bad_df):
    if is_instance(bad_df, dd, "DataFrame"):
        pytest.xfail("Dask DataFrames cannot handle integer column names")

    es = EntitySet(id="Failure")
    error_text = r"All column names must be strings \(Columns \[3\] are not strings\)"
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_name="str_cols", dataframe=bad_df, index="a")

    bad_df.ww.init()
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(dataframe_name="str_cols", dataframe=bad_df)


def test_sort_time_id():
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s")[
                ::-1
            ],
        },
    )

    es = EntitySet(
        "test",
        dataframes={"t": (transactions_df.copy(), "id", "transaction_time")},
    )
    assert es["t"] is not transactions_df
    times = list(es["t"].transaction_time)
    assert times == sorted(list(transactions_df.transaction_time))


def test_already_sorted_parameter():
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "transaction_time": [
                datetime(2014, 4, 6),
                datetime(2012, 4, 8),
                datetime(2012, 4, 8),
                datetime(2013, 4, 8),
                datetime(2015, 4, 8),
                datetime(2016, 4, 9),
            ],
        },
    )

    es = EntitySet(id="test")
    es.add_dataframe(
        transactions_df.copy(),
        dataframe_name="t",
        index="id",
        time_index="transaction_time",
        already_sorted=True,
    )

    assert es["t"] is not transactions_df
    times = list(es["t"].transaction_time)
    assert times == list(transactions_df.transaction_time)


def test_concat_not_inplace(es):
    first_es = copy.deepcopy(es)
    for df in first_es.dataframes:
        new_df = df.loc[[], :]
        first_es.replace_dataframe(df.ww.name, new_df)

    second_es = copy.deepcopy(es)

    # set the data description
    first_es.metadata

    new_es = first_es.concat(second_es)

    assert new_es == es
    assert new_es._data_description is None
    assert first_es._data_description is not None


def test_concat_inplace(es):
    first_es = copy.deepcopy(es)
    second_es = copy.deepcopy(es)
    for df in first_es.dataframes:
        new_df = df.loc[[], :]
        first_es.replace_dataframe(df.ww.name, new_df)

    # set the data description
    es.metadata

    es.concat(first_es, inplace=True)

    assert second_es == es
    assert es._data_description is None


def test_concat_with_lti(es):
    first_es = copy.deepcopy(es)
    for df in first_es.dataframes:
        if first_es.dataframe_type == Library.SPARK:
            # Spark cannot compute last time indexes on an empty Dataframe
            new_df = df.head(1)
        else:
            new_df = df.loc[[], :]
        first_es.replace_dataframe(df.ww.name, new_df)

    second_es = copy.deepcopy(es)

    first_es.add_last_time_indexes()
    second_es.add_last_time_indexes()
    es.add_last_time_indexes()

    new_es = first_es.concat(second_es)

    assert new_es == es

    first_es["stores"].ww.pop(LTI_COLUMN_NAME)
    first_es["stores"].ww.metadata.pop("last_time_index")
    second_es["stores"].ww.pop(LTI_COLUMN_NAME)
    second_es["stores"].ww.metadata.pop("last_time_index")

    assert not first_es.__eq__(es, deep=False)
    assert not second_es.__eq__(es, deep=False)
    assert LTI_COLUMN_NAME not in first_es["stores"]
    assert LTI_COLUMN_NAME not in second_es["stores"]

    new_es = first_es.concat(second_es)

    assert new_es.__eq__(es, deep=True)
    # stores will get last time index re-added because it has children that will get lti calculated
    assert LTI_COLUMN_NAME in new_es["stores"]


def test_concat_errors(es):
    # entitysets are not equal
    copy_es = copy.deepcopy(es)
    copy_es["customers"].ww.pop("phone_number")

    error = (
        "Entitysets must have the same dataframes, relationships" ", and column names"
    )
    with pytest.raises(ValueError, match=error):
        es.concat(copy_es)


def test_concat_sort_index_with_time_index(pd_es):
    # only pandas dataframes sort on the index and time index
    es1 = copy.deepcopy(pd_es)
    es1.replace_dataframe(
        dataframe_name="customers",
        df=pd_es["customers"].loc[[0, 1], :],
        already_sorted=True,
    )
    es2 = copy.deepcopy(pd_es)
    es2.replace_dataframe(
        dataframe_name="customers",
        df=pd_es["customers"].loc[[2], :],
        already_sorted=True,
    )

    combined_es_order_1 = es1.concat(es2)
    combined_es_order_2 = es2.concat(es1)

    assert list(combined_es_order_1["customers"].index) == [2, 0, 1]
    assert list(combined_es_order_2["customers"].index) == [2, 0, 1]
    assert combined_es_order_1.__eq__(pd_es, deep=True)
    assert combined_es_order_2.__eq__(pd_es, deep=True)
    assert combined_es_order_2.__eq__(combined_es_order_1, deep=True)


def test_concat_sort_index_without_time_index(pd_es):
    # Sorting is only performed on DataFrames with time indices
    es1 = copy.deepcopy(pd_es)
    es1.replace_dataframe(
        dataframe_name="products",
        df=pd_es["products"].iloc[[0, 1, 2], :],
        already_sorted=True,
    )
    es2 = copy.deepcopy(pd_es)
    es2.replace_dataframe(
        dataframe_name="products",
        df=pd_es["products"].iloc[[3, 4, 5], :],
        already_sorted=True,
    )

    combined_es_order_1 = es1.concat(es2)
    combined_es_order_2 = es2.concat(es1)

    # order matters when we don't sort
    assert list(combined_es_order_1["products"].index) == [
        "Haribo sugar-free gummy bears",
        "car",
        "toothpaste",
        "brown bag",
        "coke zero",
        "taco clock",
    ]
    assert list(combined_es_order_2["products"].index) == [
        "brown bag",
        "coke zero",
        "taco clock",
        "Haribo sugar-free gummy bears",
        "car",
        "toothpaste",
    ]
    assert combined_es_order_1.__eq__(pd_es, deep=True)
    assert not combined_es_order_2.__eq__(pd_es, deep=True)
    assert combined_es_order_2.__eq__(pd_es, deep=False)
    assert not combined_es_order_2.__eq__(combined_es_order_1, deep=True)


def test_concat_with_make_index(es):
    df = pd.DataFrame({"id": [0, 1, 2], "category": ["a", "b", "a"]})
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=2)
    elif es.dataframe_type == Library.SPARK:
        df = ps.from_pandas(df)
    logical_types = {"id": Categorical, "category": Categorical}
    es.add_dataframe(
        dataframe=df,
        dataframe_name="test_df",
        index="id1",
        make_index=True,
        logical_types=logical_types,
    )

    es_1 = copy.deepcopy(es)
    es_2 = copy.deepcopy(es)

    assert es.__eq__(es_1, deep=True)
    assert es.__eq__(es_2, deep=True)

    # map of what rows to take from es_1 and es_2 for each dataframe
    emap = {
        "log": [list(range(10)) + [14, 15, 16], list(range(10, 14)) + [15, 16]],
        "sessions": [[0, 1, 2], [1, 3, 4, 5]],
        "customers": [[0, 2], [1, 2]],
        "test_df": [[0, 1], [0, 2]],
    }

    for i, _es in enumerate([es_1, es_2]):
        for df_name, rows in emap.items():
            df = _es[df_name]
            _es.replace_dataframe(dataframe_name=df_name, df=df.loc[rows[i]])

    assert es.__eq__(es_1, deep=False)
    assert es.__eq__(es_2, deep=False)
    if es.dataframe_type == Library.PANDAS:
        assert not es.__eq__(es_1, deep=True)
        assert not es.__eq__(es_2, deep=True)

    old_es_1 = copy.deepcopy(es_1)
    old_es_2 = copy.deepcopy(es_2)
    es_3 = es_1.concat(es_2)

    assert old_es_1.__eq__(es_1, deep=True)
    assert old_es_2.__eq__(es_2, deep=True)

    assert es_3.__eq__(es, deep=True)


@pytest.fixture
def pd_transactions_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "fraud": [True, False, False, False, True, True],
        },
    )


@pytest.fixture
def dd_transactions_df(pd_transactions_df):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_transactions_df, npartitions=3)


@pytest.fixture
def spark_transactions_df(pd_transactions_df):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_transactions_df)


@pytest.fixture(
    params=["pd_transactions_df", "dd_transactions_df", "spark_transactions_df"],
)
def transactions_df(request):
    return request.getfixturevalue(request.param)


def test_set_time_type_on_init(transactions_df):
    # create cards dataframe
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    if is_instance(transactions_df, dd, "DataFrame"):
        cards_df = dd.from_pandas(cards_df, npartitions=3)
    if ps and isinstance(transactions_df, ps.DataFrame):
        cards_df = ps.from_pandas(cards_df)
    if not isinstance(transactions_df, pd.DataFrame):
        cards_logical_types = {"id": Categorical}
        transactions_logical_types = {
            "id": Integer,
            "card_id": Categorical,
            "transaction_time": Integer,
            "fraud": Boolean,
        }
    else:
        cards_logical_types = None
        transactions_logical_types = None

    dataframes = {
        "cards": (cards_df, "id", None, cards_logical_types),
        "transactions": (
            transactions_df,
            "id",
            "transaction_time",
            transactions_logical_types,
        ),
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    es = EntitySet("fraud", dataframes, relationships)
    # assert time_type is set
    assert es.time_type == "numeric"


def test_sets_time_when_adding_dataframe(transactions_df):
    accounts_df = pd.DataFrame(
        {
            "id": [3, 4, 5],
            "signup_date": [
                datetime(2002, 5, 1),
                datetime(2006, 3, 20),
                datetime(2011, 11, 11),
            ],
        },
    )
    accounts_df_string = pd.DataFrame(
        {"id": [3, 4, 5], "signup_date": ["element", "exporting", "editable"]},
    )
    if is_instance(transactions_df, dd, "DataFrame"):
        accounts_df = dd.from_pandas(accounts_df, npartitions=2)
    if ps and isinstance(transactions_df, ps.DataFrame):
        accounts_df = ps.from_pandas(accounts_df)
    if not isinstance(transactions_df, pd.DataFrame):
        accounts_logical_types = {"id": Categorical, "signup_date": Datetime}
        transactions_logical_types = {
            "id": Integer,
            "card_id": Categorical,
            "transaction_time": Integer,
            "fraud": Boolean,
        }
    else:
        accounts_logical_types = None
        transactions_logical_types = None

    # create empty entityset
    es = EntitySet("fraud")
    # assert it's not set
    assert getattr(es, "time_type", None) is None
    # add dataframe
    es.add_dataframe(
        transactions_df,
        dataframe_name="transactions",
        index="id",
        time_index="transaction_time",
        logical_types=transactions_logical_types,
    )
    # assert time_type is set
    assert es.time_type == "numeric"
    # add another dataframe
    es.normalize_dataframe("transactions", "cards", "card_id", make_time_index=True)
    # assert time_type unchanged
    assert es.time_type == "numeric"
    # add wrong time type dataframe
    error_text = "accounts time index is Datetime type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.add_dataframe(
            accounts_df,
            dataframe_name="accounts",
            index="id",
            time_index="signup_date",
            logical_types=accounts_logical_types,
        )
    # add non time type as time index, only valid for pandas
    if isinstance(transactions_df, pd.DataFrame):
        error_text = "Time index column must contain datetime or numeric values"
        with pytest.raises(TypeError, match=error_text):
            es.add_dataframe(
                accounts_df_string,
                dataframe_name="accounts",
                index="id",
                time_index="signup_date",
            )


def test_secondary_time_index_no_primary_time_index(es):
    es["products"].ww.set_types(logical_types={"rating": "Datetime"})
    assert es["products"].ww.time_index is None

    error = (
        "Cannot set secondary time index on a DataFrame that has no primary time index."
    )
    with pytest.raises(ValueError, match=error):
        es.set_secondary_time_index("products", {"rating": ["url"]})

    assert "secondary_time_index" not in es["products"].ww.metadata
    assert es["products"].ww.time_index is None


def test_set_non_valid_time_index_type(es):
    error_text = "Time index column must be a Datetime or numeric column."
    with pytest.raises(TypeError, match=error_text):
        es["log"].ww.set_time_index("purchased")


def test_checks_time_type_setting_secondary_time_index(es):
    # entityset is timestamp time type
    assert es.time_type == Datetime
    # add secondary index that is timestamp type
    new_2nd_ti = {
        "upgrade_date": ["upgrade_date", "favorite_quote"],
        "cancel_date": ["cancel_date", "cancel_reason"],
    }
    es.set_secondary_time_index("customers", new_2nd_ti)
    assert es.time_type == Datetime
    # add secondary index that is numeric type
    new_2nd_ti = {"age": ["age", "loves_ice_cream"]}

    error_text = "customers time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index("customers", new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {"favorite_quote": ["favorite_quote", "loves_ice_cream"]}

    error_text = "customers time index not recognized as numeric or datetime"
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index("customers", new_2nd_ti)
    # add mismatched pair of secondary time indexes
    new_2nd_ti = {
        "upgrade_date": ["upgrade_date", "favorite_quote"],
        "age": ["age", "loves_ice_cream"],
    }

    error_text = "customers time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        es.set_secondary_time_index("customers", new_2nd_ti)

    # create entityset with numeric time type
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "fraud_decision_time": [11, 14, 15, 21, 22, 21],
            "transaction_city": ["City A"] * 6,
            "transaction_date": [datetime(1989, 2, i) for i in range(1, 7)],
            "fraud": [True, False, False, False, True, True],
        },
    )
    dataframes = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time"),
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    card_es = EntitySet("fraud", dataframes, relationships)
    assert card_es.time_type == "numeric"
    # add secondary index that is numeric time type
    new_2nd_ti = {"fraud_decision_time": ["fraud_decision_time", "fraud"]}
    card_es.set_secondary_time_index("transactions", new_2nd_ti)
    assert card_es.time_type == "numeric"
    # add secondary index that is timestamp type
    new_2nd_ti = {"transaction_date": ["transaction_date", "fraud"]}

    error_text = "transactions time index is Datetime type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index("transactions", new_2nd_ti)
    # add secondary index that is non-time type
    new_2nd_ti = {"transaction_city": ["transaction_city", "fraud"]}

    error_text = "transactions time index not recognized as numeric or datetime"
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index("transactions", new_2nd_ti)
    # add mixed secondary time indexes
    new_2nd_ti = {
        "transaction_city": ["transaction_city", "fraud"],
        "fraud_decision_time": ["fraud_decision_time", "fraud"],
    }
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index("transactions", new_2nd_ti)

    # add bool secondary time index
    error_text = "transactions time index not recognized as numeric or datetime"
    with pytest.raises(TypeError, match=error_text):
        card_es.set_secondary_time_index("transactions", {"fraud": ["fraud"]})


def test_normalize_dataframe(es):
    error_text = "'additional_columns' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            additional_columns="log",
        )

    error_text = "'copy_columns' must be a list, but received type.*"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            copy_columns="log",
        )

    es.normalize_dataframe(
        "sessions",
        "device_types",
        "device_type",
        additional_columns=["device_name"],
        make_time_index=False,
    )

    assert len(es.get_forward_relationships("sessions")) == 2
    assert (
        es.get_forward_relationships("sessions")[1].parent_dataframe.ww.name
        == "device_types"
    )
    assert "device_name" in es["device_types"].columns
    assert "device_name" not in es["sessions"].columns
    assert "device_type" in es["device_types"].columns


def test_normalize_dataframe_add_index_as_column(es):
    error_text = "Not adding device_type as both index and column in additional_columns"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            additional_columns=["device_name", "device_type"],
            make_time_index=False,
        )

    error_text = "Not adding device_type as both index and column in copy_columns"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            copy_columns=["device_name", "device_type"],
            make_time_index=False,
        )


def test_normalize_dataframe_new_time_index_in_base_dataframe_error_check(es):
    error_text = "'make_time_index' must be a column in the base dataframe"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            base_dataframe_name="customers",
            new_dataframe_name="cancellations",
            index="cancel_reason",
            make_time_index="non-existent",
        )


def test_normalize_dataframe_new_time_index_in_column_list_error_check(es):
    error_text = (
        "'make_time_index' must be specified in 'additional_columns' or 'copy_columns'"
    )
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            base_dataframe_name="customers",
            new_dataframe_name="cancellations",
            index="cancel_reason",
            make_time_index="cancel_date",
        )


def test_normalize_dataframe_new_time_index_copy_success_check(es):
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancellations",
        index="cancel_reason",
        make_time_index="cancel_date",
        additional_columns=[],
        copy_columns=["cancel_date"],
    )


def test_normalize_dataframe_new_time_index_additional_success_check(es):
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancellations",
        index="cancel_reason",
        make_time_index="cancel_date",
        additional_columns=["cancel_date"],
        copy_columns=[],
    )


@pytest.fixture
def pd_normalize_es():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "A": [5, 4, 2, 3],
            "time": [
                datetime(2020, 6, 3),
                (datetime(2020, 3, 12)),
                datetime(2020, 5, 1),
                datetime(2020, 4, 22),
            ],
        },
    )
    es = EntitySet("es")
    return es.add_dataframe(dataframe_name="data", dataframe=df, index="id")


@pytest.fixture
def dd_normalize_es(pd_normalize_es):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    es = EntitySet(id=pd_normalize_es.id)
    dd_df = dd.from_pandas(pd_normalize_es["data"], npartitions=2)
    dd_df.ww.init(schema=pd_normalize_es["data"].ww.schema)

    es.add_dataframe(dataframe=dd_df)
    return es


@pytest.fixture
def spark_normalize_es(pd_normalize_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    es = EntitySet(id=pd_normalize_es.id)
    spark_df = ps.from_pandas(pd_normalize_es["data"])
    spark_df.ww.init(schema=pd_normalize_es["data"].ww.schema)
    es.add_dataframe(dataframe=spark_df)
    return es


@pytest.fixture(params=["pd_normalize_es", "dd_normalize_es", "spark_normalize_es"])
def normalize_es(request):
    return request.getfixturevalue(request.param)


def test_normalize_time_index_from_none(normalize_es):
    assert normalize_es["data"].ww.time_index is None

    normalize_es.normalize_dataframe(
        base_dataframe_name="data",
        new_dataframe_name="normalized",
        index="A",
        make_time_index="time",
        copy_columns=["time"],
    )
    assert normalize_es["normalized"].ww.time_index == "time"
    df = normalize_es["normalized"]

    # only pandas sorts by time index
    if isinstance(df, pd.DataFrame):
        assert df["time"].is_monotonic_increasing


def test_raise_error_if_dupicate_additional_columns_passed(es):
    error_text = (
        "'additional_columns' contains duplicate columns. All columns must be unique."
    )
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            additional_columns=["device_name", "device_name"],
        )


def test_raise_error_if_dupicate_copy_columns_passed(es):
    error_text = (
        "'copy_columns' contains duplicate columns. All columns must be unique."
    )
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            "sessions",
            "device_types",
            "device_type",
            copy_columns=["device_name", "device_name"],
        )


def test_normalize_dataframe_copies_logical_types(es):
    es["log"].ww.set_types(
        logical_types={
            "value": Ordinal(
                order=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 15.0, 20.0],
            ),
        },
    )

    assert isinstance(es["log"].ww.logical_types["value"], Ordinal)
    assert len(es["log"].ww.logical_types["value"].order) == 10
    assert isinstance(es["log"].ww.logical_types["priority_level"], Ordinal)
    assert len(es["log"].ww.logical_types["priority_level"].order) == 3
    es.normalize_dataframe(
        "log",
        "values_2",
        "value_2",
        additional_columns=["priority_level"],
        copy_columns=["value"],
        make_time_index=False,
    )

    assert len(es.get_forward_relationships("log")) == 3
    assert es.get_forward_relationships("log")[2].parent_dataframe.ww.name == "values_2"
    assert "priority_level" in es["values_2"].columns
    assert "value" in es["values_2"].columns
    assert "priority_level" not in es["log"].columns
    assert "value" in es["log"].columns
    assert "value_2" in es["values_2"].columns
    assert isinstance(es["values_2"].ww.logical_types["priority_level"], Ordinal)
    assert len(es["values_2"].ww.logical_types["priority_level"].order) == 3
    assert isinstance(es["values_2"].ww.logical_types["value"], Ordinal)
    assert len(es["values_2"].ww.logical_types["value"].order) == 10


# sorting not supported in Dask, Spark
def test_make_time_index_keeps_original_sorting():
    trips = {
        "trip_id": [999 - i for i in range(1000)],
        "flight_time": [datetime(1997, 4, 1) for i in range(1000)],
        "flight_id": [1 for i in range(350)] + [2 for i in range(650)],
    }
    order = [i for i in range(1000)]
    df = pd.DataFrame.from_dict(trips)
    es = EntitySet("flights")
    es.add_dataframe(
        dataframe=df,
        dataframe_name="trips",
        index="trip_id",
        time_index="flight_time",
    )
    assert (es["trips"]["trip_id"] == order).all()
    es.normalize_dataframe(
        base_dataframe_name="trips",
        new_dataframe_name="flights",
        index="flight_id",
        make_time_index=True,
    )
    assert (es["trips"]["trip_id"] == order).all()


def test_normalize_dataframe_new_time_index(es):
    new_time_index = "value_time"
    es.normalize_dataframe(
        "log",
        "values",
        "value",
        make_time_index=True,
        new_dataframe_time_index=new_time_index,
    )

    assert es["values"].ww.time_index == new_time_index
    assert new_time_index in es["values"].columns
    assert len(es["values"].columns) == 2
    df = to_pandas(es["values"], sort_index=True)
    assert df[new_time_index].is_monotonic_increasing


def test_normalize_dataframe_same_index(es):
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "transaction_time": pd.date_range(start="10:00", periods=3, freq="10s"),
            "first_df_time": [1, 2, 3],
        },
    )
    es = EntitySet("example")
    es.add_dataframe(
        dataframe_name="df",
        index="id",
        time_index="transaction_time",
        dataframe=transactions_df,
    )

    error_text = "'index' must be different from the index column of the base dataframe"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            base_dataframe_name="df",
            new_dataframe_name="new_dataframe",
            index="id",
            make_time_index=True,
        )


def test_secondary_time_index(es):
    es.normalize_dataframe(
        "log",
        "values",
        "value",
        make_time_index=True,
        make_secondary_time_index={"datetime": ["comments"]},
        new_dataframe_time_index="value_time",
        new_dataframe_secondary_time_index="second_ti",
    )

    assert isinstance(es["values"].ww.logical_types["second_ti"], Datetime)
    assert es["values"].ww.semantic_tags["second_ti"] == set()
    assert es["values"].ww.metadata["secondary_time_index"] == {
        "second_ti": ["comments", "second_ti"],
    }


def test_sizeof(es):
    es.add_last_time_indexes()
    total_size = 0
    for df in es.dataframes:
        total_size += df.__sizeof__()

    assert es.__sizeof__() == total_size


def test_construct_without_id():
    assert EntitySet().id is None


def test_repr_without_id():
    match = "Entityset: None\n  DataFrames:\n  Relationships:\n    No relationships"
    assert repr(EntitySet()) == match


def test_getitem_without_id():
    error_text = "DataFrame test does not exist in entity set"
    with pytest.raises(KeyError, match=error_text):
        EntitySet()["test"]


def test_metadata_without_id():
    es = EntitySet()
    assert es.metadata.id is None


@pytest.fixture
def pd_datetime3():
    return pd.DataFrame({"id": [0, 1, 2], "ints": ["1", "2", "1"]})


@pytest.fixture
def dd_datetime3(pd_datetime3):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_datetime3, npartitions=2)


@pytest.fixture
def spark_datetime3(pd_datetime3):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_datetime3)


@pytest.fixture(params=["pd_datetime3", "dd_datetime3", "spark_datetime3"])
def datetime3(request):
    return request.getfixturevalue(request.param)


def test_datetime64_conversion(datetime3):
    df = datetime3
    df["time"] = pd.Timestamp.now()
    if ps and isinstance(df, ps.DataFrame):
        df["time"] = df["time"].astype(np.datetime64)
    else:
        df["time"] = df["time"].dt.tz_localize("UTC")

    if not isinstance(df, pd.DataFrame):
        logical_types = {"id": Integer, "ints": Integer, "time": Datetime}
    else:
        logical_types = None
    es = EntitySet(id="test")
    es.add_dataframe(
        dataframe_name="test_dataframe",
        index="id",
        dataframe=df,
        logical_types=logical_types,
    )
    es["test_dataframe"].ww.set_time_index("time")
    assert es["test_dataframe"].ww.time_index == "time"


@pytest.fixture
def pd_index_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s"),
            "first_dataframe_time": [1, 2, 3, 5, 6, 6],
        },
    )


@pytest.fixture
def dd_index_df(pd_index_df):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(pd_index_df, npartitions=3)


@pytest.fixture
def spark_index_df(pd_index_df):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(pd_index_df)


@pytest.fixture(params=["pd_index_df", "dd_index_df", "spark_index_df"])
def index_df(request):
    return request.getfixturevalue(request.param)


def test_same_index_values(index_df):
    if not isinstance(index_df, pd.DataFrame):
        logical_types = {
            "id": Integer,
            "transaction_time": Datetime,
            "first_dataframe_time": Integer,
        }
    else:
        logical_types = None

    es = EntitySet("example")

    error_text = (
        '"id" is already set as the index. An index cannot also be the time index.'
    )
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(
            dataframe_name="dataframe",
            index="id",
            time_index="id",
            dataframe=index_df,
            logical_types=logical_types,
        )

    es.add_dataframe(
        dataframe_name="dataframe",
        index="id",
        time_index="transaction_time",
        dataframe=index_df,
        logical_types=logical_types,
    )

    error_text = "time_index and index cannot be the same value, first_dataframe_time"
    with pytest.raises(ValueError, match=error_text):
        es.normalize_dataframe(
            base_dataframe_name="dataframe",
            new_dataframe_name="new_dataframe",
            index="first_dataframe_time",
            make_time_index=True,
        )


def test_use_time_index(index_df):
    if not isinstance(index_df, pd.DataFrame):
        bad_ltypes = {
            "id": Integer,
            "transaction_time": Datetime,
            "first_dataframe_time": Integer,
        }
        bad_semantic_tags = {"transaction_time": "time_index"}
        logical_types = {
            "id": Integer,
            "transaction_time": Datetime,
            "first_dataframe_time": Integer,
        }
    else:
        bad_ltypes = {"transaction_time": Datetime}
        bad_semantic_tags = {"transaction_time": "time_index"}
        logical_types = None

    es = EntitySet()

    error_text = re.escape(
        "Cannot add 'time_index' tag directly for column transaction_time. To set a column as the time index, use DataFrame.ww.set_time_index() instead.",
    )
    with pytest.raises(ValueError, match=error_text):
        es.add_dataframe(
            dataframe_name="dataframe",
            index="id",
            logical_types=bad_ltypes,
            semantic_tags=bad_semantic_tags,
            dataframe=index_df,
        )

    es.add_dataframe(
        dataframe_name="dataframe",
        index="id",
        time_index="transaction_time",
        logical_types=logical_types,
        dataframe=index_df,
    )


def test_normalize_with_datetime_time_index(es):
    es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancel_reason",
        index="cancel_reason",
        make_time_index=False,
        copy_columns=["signup_date", "upgrade_date"],
    )

    assert isinstance(es["cancel_reason"].ww.logical_types["signup_date"], Datetime)
    assert isinstance(es["cancel_reason"].ww.logical_types["upgrade_date"], Datetime)


def test_normalize_with_numeric_time_index(int_es):
    int_es.normalize_dataframe(
        base_dataframe_name="customers",
        new_dataframe_name="cancel_reason",
        index="cancel_reason",
        make_time_index=False,
        copy_columns=["signup_date", "upgrade_date"],
    )

    assert int_es["cancel_reason"].ww.semantic_tags["signup_date"] == {"numeric"}


def test_normalize_with_invalid_time_index(es):
    if es.dataframe_type == Library.DASK:
        pytest.skip(
            "Woodwork raises different error with Dask. Remove this skip once WW is updated.",
        )
    error_text = "Time index column must contain datetime or numeric values"
    with pytest.raises(TypeError, match=error_text):
        es.normalize_dataframe(
            base_dataframe_name="customers",
            new_dataframe_name="cancel_reason",
            index="cancel_reason",
            copy_columns=["upgrade_date", "favorite_quote"],
            make_time_index="favorite_quote",
        )


def test_entityset_init():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card_id": [1, 2, 1, 3, 4, 5],
            "transaction_time": [10, 12, 13, 20, 21, 20],
            "upgrade_date": [51, 23, 45, 12, 22, 53],
            "fraud": [True, False, False, False, True, True],
        },
    )
    logical_types = {"fraud": "boolean", "card_id": "integer"}
    dataframes = {
        "cards": (cards_df.copy(), "id", None, {"id": "Integer"}),
        "transactions": (
            transactions_df.copy(),
            "id",
            "transaction_time",
            logical_types,
            None,
            False,
        ),
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    es = EntitySet(id="fraud_data", dataframes=dataframes, relationships=relationships)
    assert es["transactions"].ww.index == "id"
    assert es["transactions"].ww.time_index == "transaction_time"
    es_copy = EntitySet(id="fraud_data")
    es_copy.add_dataframe(dataframe_name="cards", dataframe=cards_df.copy(), index="id")
    es_copy.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df.copy(),
        index="id",
        logical_types=logical_types,
        make_index=False,
        time_index="transaction_time",
    )
    es_copy.add_relationship("cards", "id", "transactions", "card_id")

    assert es["cards"].ww == es_copy["cards"].ww
    assert es["transactions"].ww == es_copy["transactions"].ww


def test_add_interesting_values_specified_vals(es):
    product_vals = ["coke zero", "taco clock"]
    country_vals = ["AL", "US"]
    interesting_values = {
        "product_id": product_vals,
        "countrycode": country_vals,
    }
    es.add_interesting_values(dataframe_name="log", values=interesting_values)

    assert es["log"].ww["product_id"].ww.metadata["interesting_values"] == product_vals
    assert es["log"].ww["countrycode"].ww.metadata["interesting_values"] == country_vals


def test_add_interesting_values_vals_specified_without_dataframe_name(es):
    interesting_values = {
        "countrycode": ["AL", "US"],
    }
    error_msg = "dataframe_name must be specified if values are provided"
    with pytest.raises(ValueError, match=error_msg):
        es.add_interesting_values(values=interesting_values)


def test_add_interesting_values_single_dataframe(pd_es):
    pd_es.add_interesting_values(dataframe_name="log")

    expected_vals = {
        "zipcode": ["02116", "02116-3899", "12345-6789", "1234567890", "0"],
        "countrycode": ["US", "AL", "ALB", "USA"],
        "subregioncode": ["US-AZ", "US-MT", "ZM-06", "UG-219"],
        "priority_level": [0, 1, 2],
    }

    for col in pd_es["log"].columns:
        if col in expected_vals:
            assert (
                pd_es["log"].ww.columns[col].metadata.get("interesting_values")
                == expected_vals[col]
            )
        else:
            assert (
                pd_es["log"].ww.columns[col].metadata.get("interesting_values") is None
            )


def test_add_interesting_values_multiple_dataframes(pd_es):
    pd_es.add_interesting_values()
    expected_cols_with_vals = {
        "régions": {"language"},
        "stores": {},
        "products": {"department"},
        "customers": {"cancel_reason", "engagement_level"},
        "sessions": {"device_type", "device_name"},
        "log": {"zipcode", "countrycode", "subregioncode", "priority_level"},
        "cohorts": {"cohort_name"},
    }
    for df_id, df in pd_es.dataframe_dict.items():
        expected_cols = expected_cols_with_vals[df_id]
        for col in df.columns:
            if col in expected_cols:
                assert df.ww.columns[col].metadata.get("interesting_values") is not None
            else:
                assert df.ww.columns[col].metadata.get("interesting_values") is None


def test_add_interesting_values_verbose_output(caplog):
    es = load_retail(nrows=200)
    es["order_products"].ww.set_types({"quantity": "Categorical"})
    es["orders"].ww.set_types({"country": "Categorical"})
    logger = logging.getLogger("featuretools")
    logger.propagate = True
    logger_es = logging.getLogger("featuretools.entityset")
    logger_es.propagate = True
    es.add_interesting_values(verbose=True, max_values=10)
    logger.propagate = False
    logger_es.propagate = False
    assert (
        "Column country: Marking United Kingdom as an interesting value" in caplog.text
    )
    assert "Column quantity: Marking 6 as an interesting value" in caplog.text


def test_entityset_equality(es):
    first_es = EntitySet()
    second_es = EntitySet()
    assert first_es == second_es

    first_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        time_index="signup_date",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )
    assert first_es != second_es

    second_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )
    assert first_es != second_es

    first_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )
    second_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        time_index="signup_date",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )
    assert first_es == second_es

    first_es.add_relationship("customers", "id", "sessions", "customer_id")
    assert first_es != second_es
    assert second_es != first_es

    second_es.add_relationship("customers", "id", "sessions", "customer_id")
    assert first_es == second_es


def test_entityset_dataframe_dict_and_relationship_equality(es):
    first_es = EntitySet()
    second_es = EntitySet()

    first_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )

    # Tests if two entity sets are not equal if they have a different
    # number of dataframes attached.
    # first_es has 1 dataframe, second_es has 0 dataframes attached.
    assert first_es != second_es

    second_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )

    # Tests if two entity sets are not equal if they have a different
    # dataframes attached.
    # first_es has the sessions dataframe attached,
    # second_es has the customers dataframe attached.
    assert first_es != second_es

    first_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )
    first_es.add_dataframe(
        dataframe_name="stores",
        dataframe=es["stores"].copy(),
        index="id",
        logical_types=es["stores"].ww.logical_types,
        semantic_tags=get_df_tags(es["stores"]),
    )
    first_es.add_dataframe(
        dataframe_name="régions",
        dataframe=es["régions"].copy(),
        index="id",
        logical_types=es["régions"].ww.logical_types,
        semantic_tags=get_df_tags(es["régions"]),
    )

    second_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )
    second_es.add_dataframe(
        dataframe_name="stores",
        dataframe=es["stores"].copy(),
        index="id",
        logical_types=es["stores"].ww.logical_types,
        semantic_tags=get_df_tags(es["stores"]),
    )
    second_es.add_dataframe(
        dataframe_name="régions",
        dataframe=es["régions"].copy(),
        index="id",
        logical_types=es["régions"].ww.logical_types,
        semantic_tags=get_df_tags(es["régions"]),
    )

    # Now the two entity sets should be equal,
    # since they have the same dataframes.
    assert first_es == second_es

    first_es.add_relationship("customers", "id", "sessions", "customer_id")
    second_es.add_relationship("régions", "id", "stores", "région_id")

    # Test if two entity sets are not equal
    # if they have different relationships.
    assert first_es != second_es


def test_entityset_id_equality():
    first_es = EntitySet(id="first")
    first_es_copy = EntitySet(id="first")
    second_es = EntitySet(id="second")

    assert first_es != second_es
    assert first_es == first_es_copy


def test_entityset_time_type_equality():
    first_es = EntitySet()
    second_es = EntitySet()
    assert first_es == second_es

    first_es.time_type = "numeric"
    assert first_es != second_es

    second_es.time_type = Datetime
    assert first_es != second_es

    second_es.time_type = "numeric"
    assert first_es == second_es


def test_entityset_deep_equality(es):
    first_es = EntitySet()
    second_es = EntitySet()

    first_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        time_index="signup_date",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )
    first_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )

    second_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=es["sessions"].copy(),
        index="id",
        logical_types=es["sessions"].ww.logical_types,
        semantic_tags=get_df_tags(es["sessions"]),
    )
    second_es.add_dataframe(
        dataframe_name="customers",
        dataframe=es["customers"].copy(),
        index="id",
        time_index="signup_date",
        logical_types=es["customers"].ww.logical_types,
        semantic_tags=get_df_tags(es["customers"]),
    )

    assert first_es.__eq__(second_es, deep=False)
    assert first_es.__eq__(second_es, deep=True)

    # Woodwork metadata only gets included in deep equality check
    first_es["sessions"].ww.metadata["created_by"] = "user0"

    assert first_es.__eq__(second_es, deep=False)
    assert not first_es.__eq__(second_es, deep=True)

    second_es["sessions"].ww.metadata["created_by"] = "user0"

    assert first_es.__eq__(second_es, deep=False)
    assert first_es.__eq__(second_es, deep=True)

    updated_df = first_es["customers"].loc[[2, 0], :]
    first_es.replace_dataframe("customers", updated_df)

    assert first_es.__eq__(second_es, deep=False)
    # Uses woodwork equality which only looks at df content for pandas
    if isinstance(updated_df, pd.DataFrame):
        assert not first_es.__eq__(second_es, deep=True)
    else:
        assert first_es.__eq__(second_es, deep=True)


@pytest.fixture(params=["make_es", "dask_es_to_copy"])
def es_to_copy(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dask_es_to_copy(make_es):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    es = EntitySet(id=make_es.id)
    for df in make_es.dataframes:
        dd_df = dd.from_pandas(df.reset_index(drop=True), npartitions=4)
        dd_df.ww.init(schema=df.ww.schema)
        es.add_dataframe(dd_df)

    for rel in make_es.relationships:
        es.add_relationship(
            rel.parent_dataframe.ww.name,
            rel._parent_column_name,
            rel.child_dataframe.ww.name,
            rel._child_column_name,
        )
    return es


def test_deepcopy_entityset(es_to_copy):
    # Uses make_es since the es fixture uses deepcopy
    copied_es = copy.deepcopy(es_to_copy)

    assert copied_es == es_to_copy
    assert copied_es is not es_to_copy

    for df_name in es_to_copy.dataframe_dict.keys():
        original_df = es_to_copy[df_name]
        new_df = copied_es[df_name]

        assert new_df.ww.schema == original_df.ww.schema
        assert new_df.ww._schema is not original_df.ww._schema

        pd.testing.assert_frame_equal(to_pandas(new_df), to_pandas(original_df))
        assert new_df is not original_df


def test_deepcopy_entityset_woodwork_changes(es):
    copied_es = copy.deepcopy(es)

    assert copied_es == es
    assert copied_es is not es

    copied_es["products"].ww.add_semantic_tags({"id": "new_tag"})

    assert copied_es["products"].ww.semantic_tags["id"] == {"index", "new_tag"}
    assert es["products"].ww.semantic_tags["id"] == {"index"}
    assert copied_es != es


def test_deepcopy_entityset_featuretools_changes(es):
    copied_es = copy.deepcopy(es)

    assert copied_es == es
    assert copied_es is not es

    copied_es.set_secondary_time_index(
        "customers",
        {"upgrade_date": ["engagement_level"]},
    )
    assert copied_es["customers"].ww.metadata["secondary_time_index"] == {
        "upgrade_date": ["engagement_level", "upgrade_date"],
    }
    assert es["customers"].ww.metadata["secondary_time_index"] == {
        "cancel_date": ["cancel_reason", "cancel_date"],
    }


def test_dataframe_type_empty_es():
    es = EntitySet("test")
    assert es.dataframe_type is None


def test_dataframe_type_pandas_es(pd_es):
    assert pd_es.dataframe_type == Library.PANDAS


def test_es__getstate__key_unique(es):
    assert not hasattr(es, WW_SCHEMA_KEY)


def test_pd_es_pickling(pd_es):
    pkl = pickle.dumps(pd_es)
    unpickled = pickle.loads(pkl)

    assert pd_es.__eq__(unpickled, deep=True)
    assert not hasattr(unpickled, WW_SCHEMA_KEY)


def test_empty_es_pickling():
    es = EntitySet(id="empty")
    pkl = pickle.dumps(es)
    unpickled = pickle.loads(pkl)

    assert es.__eq__(unpickled, deep=True)


@patch("featuretools.entityset.entityset.EntitySet.add_dataframe")
def test_setitem(add_dataframe):
    es = EntitySet()
    df = pd.DataFrame()
    es["new_df"] = df
    assert add_dataframe.called
    add_dataframe.assert_called_with(dataframe=df, dataframe_name="new_df")


def test_latlong_nan_normalization(latlong_df):
    latlong_df.ww.init(
        name="latLong",
        index="idx",
        logical_types={"latLong": "LatLong"},
    )

    dataframes = {"latLong": (latlong_df,)}

    relationships = []

    es = EntitySet("latlong-test", dataframes, relationships)

    normalized_df = to_pandas(es["latLong"], sort_index=True)

    expected_df = pd.DataFrame(
        {"idx": [0, 1, 2], "latLong": [(np.nan, np.nan), (1, 2), (np.nan, np.nan)]},
    )

    pd.testing.assert_frame_equal(normalized_df, expected_df)


def test_latlong_nan_normalization_add_dataframe(latlong_df):
    latlong_df.ww.init(
        name="latLong",
        index="idx",
        logical_types={"latLong": "LatLong"},
    )

    es = EntitySet("latlong-test")

    es.add_dataframe(latlong_df)

    normalized_df = to_pandas(es["latLong"], sort_index=True)

    expected_df = pd.DataFrame(
        {"idx": [0, 1, 2], "latLong": [(np.nan, np.nan), (1, 2), (np.nan, np.nan)]},
    )

    pd.testing.assert_frame_equal(normalized_df, expected_df)
