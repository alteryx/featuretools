from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from woodwork.exceptions import TypeConversionError
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
)

from featuretools.entityset.entityset import LTI_COLUMN_NAME, EntitySet
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library, import_or_none, is_instance

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def test_empty_es():
    es = EntitySet("es")
    assert es.id == "es"
    assert es.dataframe_dict == {}
    assert es.relationships == []
    assert es.time_type is None


@pytest.fixture
def pd_df():
    return pd.DataFrame({"id": [0, 1, 2], "category": ["a", "b", "c"]}).astype(
        {"category": "category"},
    )


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


def test_init_es_with_dataframe(df):
    es = EntitySet("es", dataframes={"table": (df, "id")})
    assert es.id == "es"
    assert len(es.dataframe_dict) == 1
    assert es["table"] is df

    assert es["table"].ww.schema is not None
    assert isinstance(es["table"].ww.logical_types["id"], Integer)
    assert isinstance(es["table"].ww.logical_types["category"], Categorical)


def test_init_es_with_woodwork_table_same_name(df):
    df.ww.init(index="id", name="table")
    es = EntitySet("es", dataframes={"table": (df,)})

    assert es.id == "es"
    assert len(es.dataframe_dict) == 1
    assert es["table"] is df

    assert es["table"].ww.schema is not None

    assert es["table"].ww.index == "id"
    assert es["table"].ww.time_index is None

    assert isinstance(es["table"].ww.logical_types["id"], Integer)
    assert isinstance(es["table"].ww.logical_types["category"], Categorical)


def test_init_es_with_woodwork_table_diff_name_error(df):
    df.ww.init(index="id", name="table")
    error = "Naming conflict in dataframes dictionary: dictionary key 'diff_name' does not match dataframe name 'table'"
    with pytest.raises(ValueError, match=error):
        EntitySet("es", dataframes={"diff_name": (df,)})


def test_init_es_with_dataframe_and_params(df):
    logical_types = {"id": "NaturalLanguage", "category": NaturalLanguage}
    semantic_tags = {"category": "new_tag"}
    es = EntitySet(
        "es",
        dataframes={"table": (df, "id", None, logical_types, semantic_tags)},
    )

    assert es.id == "es"
    assert len(es.dataframe_dict) == 1
    assert es["table"] is df

    assert es["table"].ww.schema is not None

    assert es["table"].ww.index == "id"
    assert es["table"].ww.time_index is None

    assert isinstance(es["table"].ww.logical_types["id"], NaturalLanguage)
    assert isinstance(es["table"].ww.logical_types["category"], NaturalLanguage)

    assert es["table"].ww.semantic_tags["id"] == {"index"}
    assert es["table"].ww.semantic_tags["category"] == {"new_tag"}


def test_init_es_with_multiple_dataframes(pd_df):
    second_df = pd.DataFrame({"id": [0, 1, 2, 3], "first_table_id": [1, 2, 2, 1]})

    pd_df.ww.init(name="first_table", index="id")

    es = EntitySet(
        "es",
        dataframes={
            "first_table": (pd_df,),
            "second_table": (
                second_df,
                "id",
                None,
                None,
                {"first_table_id": "foreign_key"},
            ),
        },
    )

    assert len(es.dataframe_dict) == 2
    assert es["first_table"].ww.schema is not None
    assert es["second_table"].ww.schema is not None


def test_add_dataframe_to_es(df):
    es1 = EntitySet("es")
    assert es1.dataframe_dict == {}
    es1.add_dataframe(
        df,
        dataframe_name="table",
        index="id",
        semantic_tags={"category": "new_tag"},
    )
    assert len(es1.dataframe_dict) == 1

    copy_df = df.ww.copy()

    es2 = EntitySet("es")
    assert es2.dataframe_dict == {}
    es2.add_dataframe(copy_df)
    assert len(es2.dataframe_dict) == 1

    assert es1["table"].ww == es2["table"].ww


def test_change_es_dataframe_schema(df):
    df.ww.init(index="id", name="table")
    es = EntitySet("es", dataframes={"table": (df,)})

    assert es["table"].ww.index == "id"

    es["table"].ww.set_index("category")
    assert es["table"].ww.index == "category"


def test_init_es_with_relationships(pd_df):
    second_df = pd.DataFrame({"id": [0, 1, 2, 3], "first_table_id": [1, 2, 2, 1]})

    pd_df.ww.init(name="first_table", index="id")
    second_df.ww.init(name="second_table", index="id")

    es = EntitySet(
        "es",
        dataframes={"first_table": (pd_df,), "second_table": (second_df,)},
        relationships=[("first_table", "id", "second_table", "first_table_id")],
    )

    assert len(es.relationships) == 1

    forward_dataframes = [name for name, _ in es.get_forward_dataframes("second_table")]
    assert forward_dataframes[0] == "first_table"

    relationship = es.relationships[0]
    assert "foreign_key" in relationship.child_column.ww.semantic_tags
    assert "index" in relationship.parent_column.ww.semantic_tags


@pytest.fixture
def dates_df():
    return pd.DataFrame(
        {
            "backwards_order": [8, 7, 6, 5, 4, 3, 2, 1, 0],
            "dates_backwards": [
                "2020-09-09",
                "2020-09-08",
                "2020-09-07",
                "2020-09-06",
                "2020-09-05",
                "2020-09-04",
                "2020-09-03",
                "2020-09-02",
                "2020-09-01",
            ],
            "random_order": [7, 6, 8, 0, 2, 4, 3, 1, 5],
            "repeating_dates": [
                "2020-08-01",
                "2019-08-01",
                "2020-08-01",
                "2012-08-01",
                "2019-08-01",
                "2019-08-01",
                "2019-08-01",
                "2013-08-01",
                "2019-08-01",
            ],
            "special": [7, 8, 0, 1, 4, 2, 6, 3, 5],
            "special_dates": [
                "2020-08-01",
                "2019-08-01",
                "2020-08-01",
                "2012-08-01",
                "2019-08-01",
                "2019-08-01",
                "2019-08-01",
                "2013-08-01",
                "2019-08-01",
            ],
        },
    )


def test_add_secondary_time_index(dates_df):
    dates_df.ww.init(
        name="dates_table",
        index="backwards_order",
        time_index="dates_backwards",
    )
    es = EntitySet("es")
    es.add_dataframe(
        dates_df,
        secondary_time_index={"repeating_dates": ["random_order", "special"]},
    )

    assert dates_df.ww.metadata["secondary_time_index"] == {
        "repeating_dates": ["random_order", "special", "repeating_dates"],
    }


def test_time_type_check_order(dates_df):
    dates_df.ww.init(
        name="dates_table",
        index="backwards_order",
        time_index="random_order",
    )
    es = EntitySet("es")

    error = "dates_table time index is Datetime type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error):
        es.add_dataframe(
            dates_df,
            secondary_time_index={"repeating_dates": ["random_order", "special"]},
        )

    assert "secondary_time_index" not in dates_df.ww.metadata


def test_add_time_index_through_woodwork_different_type(dates_df):
    dates_df.ww.init(
        name="dates_table",
        index="backwards_order",
        time_index="dates_backwards",
    )
    es = EntitySet("es")

    es.add_dataframe(
        dates_df,
        secondary_time_index={"repeating_dates": ["random_order", "special"]},
    )

    assert dates_df.ww.metadata["secondary_time_index"] == {
        "repeating_dates": ["random_order", "special", "repeating_dates"],
    }
    assert es.time_type == Datetime

    assert es._check_uniform_time_index(es["dates_table"]) is None

    dates_df.ww.set_time_index("random_order")
    assert dates_df.ww.time_index == "random_order"

    error = "dates_table time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error):
        es._check_uniform_time_index(es["dates_table"])


def test_init_with_mismatched_time_types(dates_df):
    dates_df.ww.init(
        name="dates_table",
        index="backwards_order",
        time_index="repeating_dates",
    )
    es = EntitySet("es")
    es.add_dataframe(dates_df, secondary_time_index={"special_dates": ["special"]})
    assert es.time_type == Datetime

    nums_df = pd.DataFrame({"id": [1, 2, 3], "times": [9, 8, 7]})
    nums_df.ww.init(name="numerics_table", index="id", time_index="times")

    error = "numerics_table time index is numeric type which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=error):
        es.add_dataframe(nums_df)


def test_int_double_time_type(dates_df):
    dates_df.ww.init(
        name="dates_table",
        index="backwards_order",
        time_index="random_order",
        logical_types={"random_order": "Integer", "special": "Double"},
    )
    es = EntitySet("es")

    # Both random_order and special are numeric, but they are different logical types
    es.add_dataframe(dates_df, secondary_time_index={"special": ["dates_backwards"]})

    assert isinstance(es["dates_table"].ww.logical_types["random_order"], Integer)
    assert isinstance(es["dates_table"].ww.logical_types["special"], Double)

    assert es["dates_table"].ww.time_index == "random_order"
    assert "special" in es["dates_table"].ww.metadata["secondary_time_index"]


def test_normalize_dataframe():
    df = pd.DataFrame(
        {
            "id": range(4),
            "full_name": [
                "Mr. John Doe",
                "Doe, Mrs. Jane",
                "James Brown",
                "Ms. Paige Turner",
            ],
            "email": [
                "john.smith@example.com",
                np.nan,
                "team@featuretools.com",
                "junk@example.com",
            ],
            "phone_number": [
                "5555555555",
                "555-555-5555",
                "1-(555)-555-5555",
                "555-555-5555",
            ],
            "age": pd.Series([33, None, 33, 57], dtype="Int64"),
            "signup_date": [pd.to_datetime("2020-09-01")] * 4,
            "is_registered": pd.Series([True, False, True, None], dtype="boolean"),
        },
    )

    df.ww.init(name="first_table", index="id", time_index="signup_date")
    es = EntitySet("es")
    es.add_dataframe(df)
    es.normalize_dataframe(
        "first_table",
        "second_table",
        "age",
        additional_columns=["phone_number", "full_name"],
        make_time_index=True,
    )
    assert len(es.dataframe_dict) == 2
    assert "foreign_key" in es["first_table"].ww.semantic_tags["age"]


def test_replace_dataframe():
    df = pd.DataFrame(
        {
            "id": range(4),
            "full_name": [
                "Mr. John Doe",
                "Doe, Mrs. Jane",
                "James Brown",
                "Ms. Paige Turner",
            ],
            "email": [
                "john.smith@example.com",
                np.nan,
                "team@featuretools.com",
                "junk@example.com",
            ],
            "phone_number": [
                "5555555555",
                "555-555-5555",
                "1-(555)-555-5555",
                "555-555-5555",
            ],
            "age": pd.Series([33, None, 33, 57], dtype="Int64"),
            "signup_date": [pd.to_datetime("2020-09-01")] * 4,
            "is_registered": pd.Series([True, False, True, None], dtype="boolean"),
        },
    )

    df.ww.init(name="table", index="id")
    es = EntitySet("es")
    es.add_dataframe(df)
    original_schema = es["table"].ww.schema

    new_df = df.iloc[2:]
    es.replace_dataframe("table", new_df)

    assert len(es["table"]) == 2
    assert es["table"].ww.schema == original_schema


def test_add_last_time_index(es):
    es.add_last_time_indexes(["products"])

    assert "last_time_index" in es["products"].ww.metadata

    assert es["products"].ww.metadata["last_time_index"] == LTI_COLUMN_NAME
    assert LTI_COLUMN_NAME in es["products"]
    assert "last_time_index" in es["products"].ww.semantic_tags[LTI_COLUMN_NAME]
    assert isinstance(es["products"].ww.logical_types[LTI_COLUMN_NAME], Datetime)


def test_add_last_time_non_numeric_index(pd_es, spark_es, dask_es):
    # Confirm that add_last_time_index works for indices that aren't numeric
    # since numeric underlying indices can accidentally match the Woodwork index
    pd_es.add_last_time_indexes(["products"])
    dask_es.add_last_time_indexes(["products"])
    spark_es.add_last_time_indexes(["products"])

    assert list(to_pandas(pd_es["products"][LTI_COLUMN_NAME]).sort_index()) == list(
        to_pandas(dask_es["products"][LTI_COLUMN_NAME]).sort_index(),
    )
    assert list(to_pandas(pd_es["products"][LTI_COLUMN_NAME]).sort_index()) == list(
        to_pandas(spark_es["products"]).sort_values("id")[LTI_COLUMN_NAME],
    )

    assert pd_es["products"].ww.schema == dask_es["products"].ww.schema
    assert pd_es["products"].ww.schema == spark_es["products"].ww.schema


def test_lti_already_has_last_time_column_name(es):
    col = es["customers"].ww.pop("loves_ice_cream")
    col.name = LTI_COLUMN_NAME

    es["customers"].ww[LTI_COLUMN_NAME] = col

    assert LTI_COLUMN_NAME in es["customers"].columns
    assert isinstance(es["customers"].ww.logical_types[LTI_COLUMN_NAME], Boolean)

    error = (
        "Cannot add a last time index on DataFrame with an existing "
        f"'{LTI_COLUMN_NAME}' column. Please rename '{LTI_COLUMN_NAME}'."
    )
    with pytest.raises(ValueError, match=error):
        es.add_last_time_indexes(["customers"])


def test_numeric_es_last_time_index_logical_type(int_es):
    assert int_es.time_type == "numeric"

    int_es.add_last_time_indexes()

    for df in int_es.dataframes:
        assert isinstance(df.ww.logical_types[LTI_COLUMN_NAME], Double)
        int_es._check_uniform_time_index(df, LTI_COLUMN_NAME)


def test_datetime_es_last_time_index_logical_type(es):
    assert es.time_type == Datetime

    es.add_last_time_indexes()

    for df in es.dataframes:
        assert isinstance(df.ww.logical_types[LTI_COLUMN_NAME], Datetime)
        es._check_uniform_time_index(df, LTI_COLUMN_NAME)


def test_dataframe_without_name(es):
    new_es = EntitySet()

    new_df = es["sessions"].copy()

    assert new_df.ww.schema is None

    error = "Cannot add dataframe to EntitySet without a name. Please provide a value for the dataframe_name parameter."
    with pytest.raises(ValueError, match=error):
        new_es.add_dataframe(new_df)


def test_dataframe_with_name_parameter(es):
    new_es = EntitySet()

    new_df = es["sessions"][["id"]]

    assert new_df.ww.schema is None

    new_es.add_dataframe(
        new_df,
        dataframe_name="df_name",
        index="id",
        logical_types={"id": "Integer"},
    )
    assert new_es["df_name"].ww.name == "df_name"


def test_woodwork_dataframe_without_name_errors(es):
    new_es = EntitySet()

    new_df = es["sessions"].ww.copy()
    new_df.ww._schema.name = None

    assert new_df.ww.name is None

    error = "Cannot add a Woodwork DataFrame to EntitySet without a name"
    with pytest.raises(ValueError, match=error):
        new_es.add_dataframe(new_df)


def test_woodwork_dataframe_with_name(es):
    new_es = EntitySet()

    new_df = es["sessions"].ww.copy()
    new_df.ww._schema.name = "df_name"

    assert new_df.ww.name == "df_name"

    new_es.add_dataframe(new_df)

    assert new_es["df_name"].ww.name == "df_name"


def test_woodwork_dataframe_ignore_conflicting_name_parameter_warning(es):
    new_es = EntitySet()

    new_df = es["sessions"].ww.copy()
    new_df.ww._schema.name = "df_name"

    assert new_df.ww.name == "df_name"

    warning = "A Woodwork-initialized DataFrame was provided, so the following parameters were ignored: dataframe_name"
    with pytest.warns(UserWarning, match=warning):
        new_es.add_dataframe(new_df, dataframe_name="conflicting_name")

    assert new_es["df_name"].ww.name == "df_name"


def test_woodwork_dataframe_same_name_parameter(es):
    new_es = EntitySet()

    new_df = es["sessions"].ww.copy()
    new_df.ww._schema.name = "df_name"

    assert new_df.ww.name == "df_name"

    new_es.add_dataframe(new_df, dataframe_name="df_name")

    assert new_es["df_name"].ww.name == "df_name"


def test_extra_woodwork_params(es):
    new_es = EntitySet()

    sessions_df = es["sessions"].ww.copy()

    assert sessions_df.ww.index == "id"
    assert sessions_df.ww.time_index is None
    assert isinstance(sessions_df.ww.logical_types["id"], Integer)

    warning_msg = (
        "A Woodwork-initialized DataFrame was provided, so the following parameters were ignored: "
        "index, time_index, logical_types, make_index, semantic_tags, already_sorted"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        new_es.add_dataframe(
            dataframe_name="sessions",
            dataframe=sessions_df,
            index="filepath",
            time_index="customer_id",
            logical_types={"id": Categorical},
            make_index=True,
            already_sorted=True,
            semantic_tags={"id": "new_tag"},
        )
    assert sessions_df.ww.index == "id"
    assert sessions_df.ww.time_index is None
    assert isinstance(sessions_df.ww.logical_types["id"], Integer)
    assert "new_tag" not in sessions_df.ww.semantic_tags


def test_replace_dataframe_errors(es):
    df = es["customers"].copy()
    if ps and isinstance(df, ps.DataFrame):
        df["new"] = [1, 2, 3]
    else:
        df["new"] = pd.Series([1, 2, 3])

    error_text = "New dataframe is missing new cohort column"
    with pytest.raises(ValueError, match=error_text):
        es.replace_dataframe(dataframe_name="customers", df=df.drop(columns=["cohort"]))

    error_text = "New dataframe contains 16 columns, expecting 15"
    with pytest.raises(ValueError, match=error_text):
        es.replace_dataframe(dataframe_name="customers", df=df)


def test_replace_dataframe_already_sorted(es):
    # test already_sorted on dataframe without time index
    df = es["sessions"].copy()
    updated_id = to_pandas(df["id"])
    updated_id.iloc[1] = 2
    updated_id.iloc[2] = 1

    df = df.set_index("id", drop=False)
    df.index.name = None

    assert es["sessions"].ww.time_index is None

    if ps and isinstance(df, ps.DataFrame):
        df["id"] = updated_id.to_list()
        df = df.sort_index()
    elif is_instance(df, dd, "DataFrame"):
        df["id"] = updated_id

    es.replace_dataframe(dataframe_name="sessions", df=df.copy(), already_sorted=False)
    sessions_df = to_pandas(es["sessions"])
    assert sessions_df["id"].iloc[1] == 2  # no sorting since time index not defined
    es.replace_dataframe(dataframe_name="sessions", df=df.copy(), already_sorted=True)
    sessions_df = to_pandas(es["sessions"])
    assert sessions_df["id"].iloc[1] == 2

    # test already_sorted on dataframe with time index
    df = es["customers"].copy()
    updated_signup = to_pandas(df["signup_date"])
    updated_signup.iloc[0] = datetime(2011, 4, 11)

    assert es["customers"].ww.time_index == "signup_date"

    if ps and isinstance(df, ps.DataFrame):
        df["signup_date"] = updated_signup.to_list()
        df = df.sort_index()
    else:
        df["signup_date"] = updated_signup

    es.replace_dataframe(dataframe_name="customers", df=df.copy(), already_sorted=True)
    customers_df = to_pandas(es["customers"])
    assert customers_df["id"].iloc[0] == 2

    # only pandas allows for sorting:
    es.replace_dataframe(dataframe_name="customers", df=df.copy(), already_sorted=False)
    updated_customers = to_pandas(es["customers"])
    if isinstance(df, pd.DataFrame):
        assert updated_customers["id"].iloc[0] == 0
    else:
        assert updated_customers["id"].iloc[0] == 2


def test_replace_dataframe_invalid_schema(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Invalid schema checks able to be caught by Woodwork only relevant for Pandas",
        )
    df = es["customers"].copy()
    df["id"] = pd.Series([1, 1, 1])

    error_text = "Index column must be unique"
    with pytest.raises(IndexError, match=error_text):
        es.replace_dataframe(dataframe_name="customers", df=df)


def test_replace_dataframe_mismatched_index(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Only pandas checks whether underlying index matches the Woodwork index",
        )
    df = es["customers"].copy()
    df["id"] = pd.Series([99, 88, 77])

    es.replace_dataframe(dataframe_name="customers", df=df)

    assert all([77, 99, 88] == es["customers"]["id"])
    assert all([77, 99, 88] == (es["customers"]["id"]).index)


def test_replace_dataframe_different_dtypes(es):
    float_dtype_df = es["customers"].copy()
    float_dtype_df = float_dtype_df.astype({"age": "float64"})

    es.replace_dataframe(dataframe_name="customers", df=float_dtype_df)

    assert es["customers"]["age"].dtype == "int64"
    assert isinstance(es["customers"].ww.logical_types["age"], Integer)

    incompatible_dtype_df = es["customers"].copy()
    incompatible_list = ["hi", "bye", "bye"]
    if ps and isinstance(incompatible_dtype_df, ps.DataFrame):
        incompatible_dtype_df["age"] = incompatible_list
    else:
        incompatible_dtype_df["age"] = pd.Series(incompatible_list)

    if isinstance(es["customers"], pd.DataFrame):
        # Dask and Spark do not error on invalid type conversion until compute
        error_msg = "Error converting datatype for age from type object to type int64. Please confirm the underlying data is consistent with logical type Integer."
        with pytest.raises(TypeConversionError, match=error_msg):
            es.replace_dataframe(dataframe_name="customers", df=incompatible_dtype_df)


@pytest.fixture()
def latlong_df_pandas():
    latlong_df = pd.DataFrame(
        {
            "tuples": pd.Series([(1, 2), (3, 4)]),
            "string_tuple": pd.Series(["(1, 2)", "(3, 4)"]),
            "bracketless_string_tuple": pd.Series(["1, 2", "3, 4"]),
            "list_strings": pd.Series([["1", "2"], ["3", "4"]]),
            "combo_tuple_types": pd.Series(["[1, 2]", "(3, 4)"]),
        },
    )
    latlong_df.set_index("string_tuple", drop=False, inplace=True)
    latlong_df.index.name = None
    return latlong_df


@pytest.fixture()
def latlong_df_dask(latlong_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(latlong_df_pandas, npartitions=2)


@pytest.fixture()
def latlong_df_spark(latlong_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(
        latlong_df_pandas.applymap(
            lambda tup: list(tup) if isinstance(tup, tuple) else tup,
        ),
    )


@pytest.fixture(params=["latlong_df_pandas", "latlong_df_dask", "latlong_df_spark"])
def latlong_df(request):
    return request.getfixturevalue(request.param)


def test_replace_dataframe_data_transformation(latlong_df):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    initial_df = latlong_df.copy()
    initial_df.ww.init(
        name="latlongs",
        index="string_tuple",
        logical_types={col_name: "LatLong" for col_name in initial_df.columns},
    )
    es = EntitySet()
    es.add_dataframe(dataframe=initial_df)

    df = to_pandas(es["latlongs"])
    expected_val = (1, 2)
    if ps and isinstance(es["latlongs"], ps.DataFrame):
        expected_val = [1, 2]
    for col in latlong_df.columns:
        series = df[col]
        assert series.iloc[0] == expected_val

    es.replace_dataframe("latlongs", latlong_df)
    df = to_pandas(es["latlongs"])
    expected_val = (3, 4)
    if ps and isinstance(es["latlongs"], ps.DataFrame):
        expected_val = [3, 4]
    for col in latlong_df.columns:
        series = df[col]
        assert series.iloc[-1] == expected_val


def test_replace_dataframe_column_order(es):
    original_column_order = es["customers"].columns.copy()

    df = es["customers"].copy()
    col = df.pop("cohort")
    df[col.name] = col

    assert not df.columns.equals(original_column_order)
    assert set(df.columns) == set(original_column_order)

    es.replace_dataframe(dataframe_name="customers", df=df)

    assert es["customers"].columns.equals(original_column_order)


def test_replace_dataframe_different_woodwork_initialized(es):
    df = es["customers"].copy()
    if ps and isinstance(df, ps.DataFrame):
        df["age"] = [1, 2, 3]
    else:
        df["age"] = pd.Series([1, 2, 3])

    # Initialize Woodwork on the new DataFrame and change the schema so it won't match the original DataFrame's schema
    df.ww.init(schema=es["customers"].ww.schema)
    df.ww.set_types(
        logical_types={"id": "NaturalLanguage", "cancel_date": "NaturalLanguage"},
    )
    assert df["id"].dtype == "string"
    assert df["cancel_date"].dtype == "string"

    assert es["customers"]["id"].dtype == "int64"
    assert es["customers"]["cancel_date"].dtype == "datetime64[ns]"

    original_schema = es["customers"].ww.schema

    warning = "Woodwork typing information on new dataframe will be replaced with existing typing information from customers"
    with pytest.warns(UserWarning, match=warning):
        es.replace_dataframe("customers", df, already_sorted=True)

    actual = to_pandas(es["customers"]["age"]).sort_values()
    assert all(actual == [1, 2, 3])

    assert es["customers"].ww._schema == original_schema
    assert es["customers"]["id"].dtype == "int64"
    assert es["customers"]["cancel_date"].dtype == "datetime64[ns]"


@pytest.mark.skipif("not dd")
def test_replace_dataframe_different_dataframe_types():
    dask_es = EntitySet(id="dask_es")

    sessions = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "user": [1, 2, 1, 3],
            "time": [
                pd.to_datetime("2019-01-10"),
                pd.to_datetime("2019-02-03"),
                pd.to_datetime("2019-01-01"),
                pd.to_datetime("2017-08-25"),
            ],
            "strings": ["I am a string", "23", "abcdef ghijk", ""],
        },
    )
    sessions_dask = dd.from_pandas(sessions, npartitions=2)
    sessions_logical_types = {
        "id": Integer,
        "user": Integer,
        "time": Datetime,
        "strings": NaturalLanguage,
    }
    sessions_semantic_tags = {"user": "foreign_key"}

    dask_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions_dask,
        index="id",
        time_index="time",
        logical_types=sessions_logical_types,
        semantic_tags=sessions_semantic_tags,
    )

    with pytest.raises(TypeError, match="Incorrect DataFrame type used"):
        dask_es.replace_dataframe("sessions", sessions)


def test_replace_dataframe_and_min_last_time_index(es):
    es.add_last_time_indexes(["products"])

    original_time_index = es["log"]["datetime"].copy()
    original_last_time_index = es["products"][LTI_COLUMN_NAME].copy()

    if ps and isinstance(original_time_index, ps.Series):
        new_time_index = ps.from_pandas(
            original_time_index.to_pandas() + pd.Timedelta(days=1),
        )
        expected_last_time_index = ps.from_pandas(
            original_last_time_index.to_pandas() + pd.Timedelta(days=1),
        )
    else:
        new_time_index = original_time_index + pd.Timedelta(days=1)
        expected_last_time_index = original_last_time_index + pd.Timedelta(days=1)

    new_dataframe = es["log"].copy()
    new_dataframe["datetime"] = new_time_index
    new_dataframe.pop(LTI_COLUMN_NAME)

    es.replace_dataframe("log", new_dataframe, recalculate_last_time_indexes=True)

    # Spark reorders indices during last time index, so we sort to confirm individual values are the same
    pd.testing.assert_series_equal(
        to_pandas(es["products"][LTI_COLUMN_NAME]).sort_index(),
        to_pandas(expected_last_time_index).sort_index(),
    )
    pd.testing.assert_series_equal(
        to_pandas(es["log"][LTI_COLUMN_NAME]).sort_index(),
        to_pandas(new_time_index).sort_index(),
        check_names=False,
    )


def test_replace_dataframe_dont_recalculate_last_time_index_present(es):
    es.add_last_time_indexes()

    original_time_index = es["customers"]["signup_date"].copy()
    original_last_time_index = es["customers"][LTI_COLUMN_NAME].copy()

    if ps and isinstance(original_time_index, ps.Series):
        new_time_index = ps.from_pandas(
            original_time_index.to_pandas() + pd.Timedelta(days=10),
        )
    else:
        new_time_index = original_time_index + pd.Timedelta(days=10)

    new_dataframe = es["customers"].copy()
    new_dataframe["signup_date"] = new_time_index

    es.replace_dataframe(
        "customers",
        new_dataframe,
        recalculate_last_time_indexes=False,
    )
    pd.testing.assert_series_equal(
        to_pandas(es["customers"][LTI_COLUMN_NAME], sort_index=True),
        to_pandas(original_last_time_index, sort_index=True),
    )


def test_replace_dataframe_dont_recalculate_last_time_index_not_present(es):
    es.add_last_time_indexes()
    original_lti_name = es["customers"].ww.metadata.get("last_time_index")
    assert original_lti_name is not None

    original_time_index = es["customers"]["signup_date"].copy()

    if ps and isinstance(original_time_index, ps.Series):
        new_time_index = ps.from_pandas(
            original_time_index.to_pandas() + pd.Timedelta(days=10),
        )
    else:
        new_time_index = original_time_index + pd.Timedelta(days=10)

    new_dataframe = es["customers"].copy()
    new_dataframe["signup_date"] = new_time_index
    new_dataframe.pop(LTI_COLUMN_NAME)

    es.replace_dataframe(
        "customers",
        new_dataframe,
        recalculate_last_time_indexes=False,
    )
    assert "last_time_index" not in es["customers"].ww.metadata
    assert original_lti_name not in es["customers"].columns


def test_replace_dataframe_recalculate_last_time_index_not_present(es):
    es.add_last_time_indexes()

    original_time_index = es["log"]["datetime"].copy()

    if ps and isinstance(original_time_index, ps.Series):
        new_time_index = ps.from_pandas(
            original_time_index.to_pandas() + pd.Timedelta(days=10),
        )
    else:
        new_time_index = original_time_index + pd.Timedelta(days=10)

    new_dataframe = es["log"].copy()
    new_dataframe["datetime"] = new_time_index
    new_dataframe.pop(LTI_COLUMN_NAME)

    es.replace_dataframe("log", new_dataframe, recalculate_last_time_indexes=True)
    pd.testing.assert_series_equal(
        to_pandas(es["log"]["datetime"]).sort_index(),
        to_pandas(new_time_index).sort_index(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        to_pandas(es["log"][LTI_COLUMN_NAME]).sort_index(),
        to_pandas(new_time_index).sort_index(),
        check_names=False,
    )


def test_replace_dataframe_recalculate_last_time_index_present(es):
    es.add_last_time_indexes()

    original_time_index = es["log"]["datetime"].copy()

    if ps and isinstance(original_time_index, ps.Series):
        new_time_index = ps.from_pandas(
            original_time_index.to_pandas() + pd.Timedelta(days=10),
        )
    else:
        new_time_index = original_time_index + pd.Timedelta(days=10)

    new_dataframe = es["log"].copy()
    new_dataframe["datetime"] = new_time_index
    assert LTI_COLUMN_NAME in new_dataframe.columns

    es.replace_dataframe("log", new_dataframe, recalculate_last_time_indexes=True)
    pd.testing.assert_series_equal(
        to_pandas(es["log"]["datetime"]).sort_index(),
        to_pandas(new_time_index).sort_index(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        to_pandas(es["log"][LTI_COLUMN_NAME]).sort_index(),
        to_pandas(new_time_index).sort_index(),
        check_names=False,
    )


def test_normalize_dataframe_loses_column_metadata(es):
    es["log"].ww.columns["value"].metadata["interesting_values"] = [0.0, 1.0]
    es["log"].ww.columns["priority_level"].metadata["interesting_values"] = [1]

    es["log"].ww.columns["value"].description = "a value column"
    es["log"].ww.columns["priority_level"].description = "a priority level column"

    assert "interesting_values" in es["log"].ww.columns["priority_level"].metadata
    assert "interesting_values" in es["log"].ww.columns["value"].metadata
    assert es["log"].ww.columns["value"].description == "a value column"
    assert (
        es["log"].ww.columns["priority_level"].description == "a priority level column"
    )

    es.normalize_dataframe(
        "log",
        "values_2",
        "value_2",
        additional_columns=["priority_level"],
        copy_columns=["value"],
        make_time_index=False,
    )

    # Metadata in the original dataframe and the new dataframe are maintained
    assert "interesting_values" in es["log"].ww.columns["value"].metadata
    assert "interesting_values" in es["values_2"].ww.columns["value"].metadata
    assert "interesting_values" in es["values_2"].ww.columns["priority_level"].metadata
    assert es["log"].ww.columns["value"].description == "a value column"
    assert es["values_2"].ww.columns["value"].description == "a value column"
    assert (
        es["values_2"].ww.columns["priority_level"].description
        == "a priority level column"
    )


def test_normalize_ww_init():
    es = EntitySet()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "col": ["a", "b", "c", "d"],
            "df2_id": [1, 1, 2, 2],
            "df2_col": [True, False, True, True],
        },
    )

    df.ww.init(index="id", name="test_name")
    es.add_dataframe(dataframe=df)

    assert es["test_name"].ww.name == "test_name"
    assert es["test_name"].ww.schema.name == "test_name"

    es.normalize_dataframe(
        "test_name",
        "new_df",
        "df2_id",
        additional_columns=["df2_col"],
    )

    assert es["test_name"].ww.name == "test_name"
    assert es["test_name"].ww.schema.name == "test_name"

    assert es["new_df"].ww.name == "new_df"
    assert es["new_df"].ww.schema.name == "new_df"
