from inspect import isclass

import numpy as np
import pandas as pd
import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
)

from featuretools import (
    AggregationFeature,
    EntitySet,
    Feature,
    IdentityFeature,
    TransformFeature,
    calculate_feature_matrix,
    dfs,
    primitives,
)
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.feature_set_calculator import (
    FeatureSetCalculator,
)
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    AddNumericScalar,
    Age,
    Count,
    Day,
    Diff,
    DiffDatetime,
    DivideByFeature,
    DivideNumeric,
    DivideNumericScalar,
    Equal,
    EqualScalar,
    FileExtension,
    First,
    FullNameToFirstName,
    FullNameToLastName,
    FullNameToTitle,
    GreaterThan,
    GreaterThanEqualTo,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    Haversine,
    Hour,
    IsIn,
    IsNull,
    Lag,
    Latitude,
    LessThan,
    LessThanEqualTo,
    LessThanEqualToScalar,
    LessThanScalar,
    Longitude,
    Min,
    Mode,
    MultiplyBoolean,
    MultiplyNumeric,
    MultiplyNumericBoolean,
    MultiplyNumericScalar,
    Not,
    NotEqual,
    NotEqualScalar,
    NumCharacters,
    NumericLag,
    NumWords,
    Percentile,
    ScalarSubtractNumericFeature,
    SubtractNumeric,
    SubtractNumericScalar,
    Sum,
    TimeSince,
    TransformPrimitive,
    get_transform_primitives,
)
from featuretools.synthesis.deep_feature_synthesis import match
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library, import_or_none
from featuretools.utils.spark_utils import pd_to_spark_clean

dd = import_or_none("dask.dataframe")


def test_init_and_name(es):
    log = es["log"]
    rating = Feature(IdentityFeature(es["products"].ww["rating"]), "log")
    log_features = [Feature(es["log"].ww[col]) for col in log.columns] + [
        Feature(rating, primitive=GreaterThanScalar(2.5)),
        Feature(rating, primitive=GreaterThanScalar(3.5)),
    ]
    # Add Timedelta feature
    # features.append(pd.Timestamp.now() - Feature(log['datetime']))
    customers_features = [
        Feature(es["customers"].ww[col]) for col in es["customers"].columns
    ]

    # check all transform primitives have a name
    for attribute_string in dir(primitives):
        attr = getattr(primitives, attribute_string)
        if isclass(attr):
            if issubclass(attr, TransformPrimitive) and attr != TransformPrimitive:
                assert getattr(attr, "name") is not None

    trans_primitives = get_transform_primitives().values()
    # If Dask EntitySet use only Dask compatible primitives
    if es.dataframe_type == Library.DASK:
        trans_primitives = [
            prim for prim in trans_primitives if Library.DASK in prim.compatibility
        ]
    if es.dataframe_type == Library.SPARK:
        trans_primitives = [
            prim for prim in trans_primitives if Library.SPARK in prim.compatibility
        ]

    for transform_prim in trans_primitives:
        # skip automated testing if a few special cases
        features_to_use = log_features
        if transform_prim in [NotEqual, Equal, FileExtension]:
            continue
        if transform_prim in [
            Age,
            FullNameToFirstName,
            FullNameToLastName,
            FullNameToTitle,
        ]:
            features_to_use = customers_features

        # use the input_types matching function from DFS
        input_types = transform_prim.input_types
        if isinstance(input_types[0], list):
            matching_inputs = match(input_types[0], features_to_use)
        else:
            matching_inputs = match(input_types, features_to_use)
        if len(matching_inputs) == 0:
            raise Exception("Transform Primitive %s not tested" % transform_prim.name)
        for prim in matching_inputs:
            instance = Feature(prim, primitive=transform_prim)

            # try to get name and calculate
            instance.get_name()
            calculate_feature_matrix([instance], entityset=es)


def test_relationship_path(es):
    f = TransformFeature(Feature(es["log"].ww["datetime"]), Hour)

    assert len(f.relationship_path) == 0


def test_serialization(es):
    value = IdentityFeature(es["log"].ww["value"])
    primitive = MultiplyNumericScalar(value=2)
    value_x2 = TransformFeature(value, primitive)

    dictionary = {
        "name": value_x2.get_name(),
        "base_features": [value.unique_name()],
        "primitive": primitive,
    }

    assert dictionary == value_x2.get_arguments()
    assert value_x2 == TransformFeature.from_dictionary(
        dictionary,
        es,
        {value.unique_name(): value},
        primitive,
    )


def test_make_trans_feat(es):
    f = Feature(es["log"].ww["datetime"], primitive=Hour)

    feature_set = FeatureSet([f])
    calculator = FeatureSetCalculator(es, feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))
    v = df[f.get_name()][0]
    assert v == 10


@pytest.fixture
def pd_simple_es():
    df = pd.DataFrame(
        {
            "id": range(4),
            "value": pd.Categorical(["a", "c", "b", "d"]),
            "value2": pd.Categorical(["a", "b", "a", "d"]),
            "object": ["time1", "time2", "time3", "time4"],
            "datetime": pd.Series(
                [
                    pd.Timestamp("2001-01-01"),
                    pd.Timestamp("2001-01-02"),
                    pd.Timestamp("2001-01-03"),
                    pd.Timestamp("2001-01-04"),
                ],
            ),
        },
    )

    es = EntitySet("equal_test")
    es.add_dataframe(dataframe_name="values", dataframe=df, index="id")

    return es


@pytest.fixture
def dd_simple_es(pd_simple_es):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dataframes = {}
    for df in pd_simple_es.dataframes:
        dataframes[df.ww.name] = (
            dd.from_pandas(df.reset_index(drop=True), npartitions=4),
            df.ww.index,
            None,
            df.ww.logical_types,
        )

    relationships = [
        (
            rel.parent_name,
            rel._parent_column_name,
            rel.child_name,
            rel._child_column_name,
        )
        for rel in pd_simple_es.relationships
    ]

    return EntitySet(
        id=pd_simple_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture
def spark_simple_es(pd_simple_es):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    dataframes = {}
    for df in pd_simple_es.dataframes:
        cleaned_df = pd_to_spark_clean(df).reset_index(drop=True)
        dataframes[df.ww.name] = (
            ps.from_pandas(cleaned_df),
            df.ww.index,
            None,
            df.ww.logical_types,
        )

    relationships = [
        (
            rel.parent_name,
            rel._parent_column_name,
            rel.child_name,
            rel._child_column_name,
        )
        for rel in pd_simple_es.relationships
    ]

    return EntitySet(
        id=pd_simple_es.id,
        dataframes=dataframes,
        relationships=relationships,
    )


@pytest.fixture(params=["pd_simple_es", "dd_simple_es", "spark_simple_es"])
def simple_es(request):
    return request.getfixturevalue(request.param)


def test_equal_categorical(simple_es):
    f1 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["value"]),
            IdentityFeature(simple_es["values"].ww["value2"]),
        ],
        primitive=Equal,
    )

    df = calculate_feature_matrix(entityset=simple_es, features=[f1])
    if simple_es.dataframe_type != Library.SPARK:
        # Spark does not support categorical dtype
        assert set(simple_es["values"]["value"].cat.categories) != set(
            simple_es["values"]["value2"].cat.categories,
        )
    assert to_pandas(df, index="id", sort_index=True)["value = value2"].to_list() == [
        True,
        False,
        False,
        True,
    ]


def test_equal_different_dtypes(simple_es):
    f1 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["object"]),
            IdentityFeature(simple_es["values"].ww["datetime"]),
        ],
        primitive=Equal,
    )
    f2 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["datetime"]),
            IdentityFeature(simple_es["values"].ww["object"]),
        ],
        primitive=Equal,
    )

    # verify that equals works for different dtypes regardless of order
    df = calculate_feature_matrix(entityset=simple_es, features=[f1, f2])

    assert to_pandas(df, index="id", sort_index=True)[
        "object = datetime"
    ].to_list() == [False, False, False, False]
    assert to_pandas(df, index="id", sort_index=True)[
        "datetime = object"
    ].to_list() == [False, False, False, False]


def test_not_equal_categorical(simple_es):
    f1 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["value"]),
            IdentityFeature(simple_es["values"].ww["value2"]),
        ],
        primitive=NotEqual,
    )

    df = calculate_feature_matrix(entityset=simple_es, features=[f1])

    if simple_es.dataframe_type != Library.SPARK:
        # Spark does not support categorical dtype
        assert set(simple_es["values"]["value"].cat.categories) != set(
            simple_es["values"]["value2"].cat.categories,
        )
    assert to_pandas(df, index="id", sort_index=True)["value != value2"].to_list() == [
        False,
        True,
        True,
        False,
    ]


def test_not_equal_different_dtypes(simple_es):
    f1 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["object"]),
            IdentityFeature(simple_es["values"].ww["datetime"]),
        ],
        primitive=NotEqual,
    )
    f2 = Feature(
        [
            IdentityFeature(simple_es["values"].ww["datetime"]),
            IdentityFeature(simple_es["values"].ww["object"]),
        ],
        primitive=NotEqual,
    )

    # verify that equals works for different dtypes regardless of order
    df = calculate_feature_matrix(entityset=simple_es, features=[f1, f2])

    assert to_pandas(df, index="id", sort_index=True)[
        "object != datetime"
    ].to_list() == [True, True, True, True]
    assert to_pandas(df, index="id", sort_index=True)[
        "datetime != object"
    ].to_list() == [True, True, True, True]


def test_diff(pd_es):
    value = Feature(pd_es["log"].ww["value"])
    customer_id_feat = Feature(pd_es["sessions"].ww["customer_id"], "log")
    diff1 = Feature(
        value,
        groupby=Feature(pd_es["log"].ww["session_id"]),
        primitive=Diff,
    )
    diff2 = Feature(value, groupby=customer_id_feat, primitive=Diff)

    feature_set = FeatureSet([diff1, diff2])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array(range(15)))

    val1 = df[diff1.get_name()].tolist()
    val2 = df[diff2.get_name()].tolist()

    correct_vals1 = [
        np.nan,
        5,
        5,
        5,
        5,
        np.nan,
        1,
        1,
        1,
        np.nan,
        np.nan,
        5,
        np.nan,
        7,
        7,
    ]
    correct_vals2 = [np.nan, 5, 5, 5, 5, -20, 1, 1, 1, -3, np.nan, 5, -5, 7, 7]
    np.testing.assert_equal(val1, correct_vals1)
    np.testing.assert_equal(val2, correct_vals2)


def test_diff_shift(pd_es):
    value = Feature(pd_es["log"].ww["value"])
    customer_id_feat = Feature(pd_es["sessions"].ww["customer_id"], "log")
    diff_periods = Feature(value, groupby=customer_id_feat, primitive=Diff(periods=1))

    feature_set = FeatureSet([diff_periods])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array(range(15)))
    val3 = df[diff_periods.get_name()].tolist()

    correct_vals3 = [np.nan, np.nan, 5, 5, 5, 5, -20, 1, 1, 1, np.nan, np.nan, 5, -5, 7]
    np.testing.assert_equal(val3, correct_vals3)


def test_diff_single_value(pd_es):
    diff = Feature(
        pd_es["stores"].ww["num_square_feet"],
        groupby=Feature(pd_es["stores"].ww["région_id"]),
        primitive=Diff,
    )
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([4]))
    assert df[diff.get_name()][4] == 6000.0


def test_diff_reordered(pd_es):
    sum_feat = Feature(
        pd_es["log"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Sum,
    )
    diff = Feature(sum_feat, primitive=Diff)
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([4, 2]))
    assert df[diff.get_name()][4] == 16
    assert df[diff.get_name()][2] == -6


def test_diff_single_value_is_nan(pd_es):
    diff = Feature(
        pd_es["stores"].ww["num_square_feet"],
        groupby=Feature(pd_es["stores"].ww["région_id"]),
        primitive=Diff,
    )
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([5]))
    assert df.shape[0] == 1
    assert df[diff.get_name()].dropna().shape[0] == 0


def test_diff_datetime(pd_es):
    diff = Feature(
        pd_es["log"].ww["datetime"],
        primitive=DiffDatetime,
    )
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array(range(15)))
    vals = pd.Series(df[diff.get_name()].tolist())
    expected_vals = pd.Series(
        [
            pd.NaT,
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=36),
            pd.Timedelta(seconds=9),
            pd.Timedelta(seconds=9),
            pd.Timedelta(seconds=9),
            pd.Timedelta(minutes=8, seconds=33),
            pd.Timedelta(days=1),
            pd.Timedelta(seconds=1),
            pd.Timedelta(seconds=59),
            pd.Timedelta(seconds=3),
            pd.Timedelta(seconds=3),
        ],
    )
    pd.testing.assert_series_equal(vals, expected_vals)


def test_diff_datetime_shift(pd_es):
    diff = Feature(
        pd_es["log"].ww["datetime"],
        primitive=DiffDatetime(periods=1),
    )
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array(range(6)))
    vals = pd.Series(df[diff.get_name()].tolist())
    expected_vals = pd.Series(
        [
            pd.NaT,
            pd.NaT,
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
            pd.Timedelta(seconds=6),
        ],
    )
    pd.testing.assert_series_equal(vals, expected_vals)


def test_compare_of_identity(es):
    to_test = [
        (EqualScalar, [False, False, True, False]),
        (NotEqualScalar, [True, True, False, True]),
        (LessThanScalar, [True, True, False, False]),
        (LessThanEqualToScalar, [True, True, True, False]),
        (GreaterThanScalar, [False, False, False, True]),
        (GreaterThanEqualToScalar, [False, False, True, True]),
    ]

    features = []
    for test in to_test:
        features.append(Feature(es["log"].ww["value"], primitive=test[0](10)))

    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=[0, 1, 2, 3],
        ),
        index="id",
        sort_index=True,
    )

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


def test_compare_of_direct(es):
    log_rating = Feature(es["products"].ww["rating"], "log")
    to_test = [
        (EqualScalar, [False, False, False, False]),
        (NotEqualScalar, [True, True, True, True]),
        (LessThanScalar, [False, False, False, True]),
        (LessThanEqualToScalar, [False, False, False, True]),
        (GreaterThanScalar, [True, True, True, False]),
        (GreaterThanEqualToScalar, [True, True, True, False]),
    ]

    features = []
    for test in to_test:
        features.append(Feature(log_rating, primitive=test[0](4.5)))

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2, 3],
    )
    df = to_pandas(df, index="id", sort_index=True)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


def test_compare_of_transform(es):
    day = Feature(es["log"].ww["datetime"], primitive=Day)
    to_test = [
        (EqualScalar, [False, True]),
        (NotEqualScalar, [True, False]),
    ]

    features = []
    for test in to_test:
        features.append(Feature(day, primitive=test[0](10)))

    df = calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 14])
    df = to_pandas(df, index="id", sort_index=True)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


def test_compare_of_agg(es):
    count_logs = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )

    to_test = [
        (EqualScalar, [False, False, False, True]),
        (NotEqualScalar, [True, True, True, False]),
        (LessThanScalar, [False, False, True, False]),
        (LessThanEqualToScalar, [False, False, True, True]),
        (GreaterThanScalar, [True, True, False, False]),
        (GreaterThanEqualToScalar, [True, True, False, True]),
    ]

    features = []
    for test in to_test:
        features.append(Feature(count_logs, primitive=test[0](2)))

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2, 3],
    )
    df = to_pandas(df, index="id", sort_index=True)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


def test_compare_all_nans(es):
    if es.dataframe_type != Library.PANDAS:
        nan_feat = Feature(
            es["log"].ww["value"],
            parent_dataframe_name="sessions",
            primitive=Min,
        )
        compare = nan_feat == 0.0
    else:
        nan_feat = Feature(
            es["log"].ww["product_id"],
            parent_dataframe_name="sessions",
            primitive=Mode,
        )
        compare = nan_feat == "brown bag"

    # before all data
    time_last = pd.Timestamp("1/1/1993")

    df = calculate_feature_matrix(
        entityset=es,
        features=[nan_feat, compare],
        instance_ids=[0, 1, 2],
        cutoff_time=time_last,
    )
    df = to_pandas(df, index="id", sort_index=True)

    assert df[nan_feat.get_name()].dropna().shape[0] == 0
    assert not df[compare.get_name()].any()


def test_arithmetic_of_val(es):
    to_test = [
        (AddNumericScalar, [2.0, 7.0, 12.0, 17.0]),
        (SubtractNumericScalar, [-2.0, 3.0, 8.0, 13.0]),
        (ScalarSubtractNumericFeature, [2.0, -3.0, -8.0, -13.0]),
        (MultiplyNumericScalar, [0, 10, 20, 30]),
        (DivideNumericScalar, [0, 2.5, 5, 7.5]),
        (DivideByFeature, [np.inf, 0.4, 0.2, 2 / 15.0]),
    ]

    features = []
    for test in to_test:
        features.append(Feature(es["log"].ww["value"], primitive=test[0](2)))

    features.append(Feature(es["log"].ww["value"]) / 0)

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2, 3],
    )
    df = to_pandas(df, index="id", sort_index=True)

    for f, test in zip(features, to_test):
        v = df[f.get_name()].tolist()
        assert v == test[1]

    test = [np.nan, np.inf, np.inf, np.inf]
    v = df[features[-1].get_name()].tolist()
    assert np.isnan(v[0])
    assert v[1:] == test[1:]


def test_arithmetic_two_vals_fails(es):
    error_text = "Not a feature"
    with pytest.raises(Exception, match=error_text):
        Feature([2, 2], primitive=AddNumeric)


def test_arithmetic_of_identity(es):
    to_test = [
        (AddNumeric, [0.0, 7.0, 14.0, 21.0]),
        (SubtractNumeric, [0, 3, 6, 9]),
        (MultiplyNumeric, [0, 10, 40, 90]),
        (DivideNumeric, [np.nan, 2.5, 2.5, 2.5]),
    ]
    # SubtractNumeric not supported for Spark EntitySets
    if es.dataframe_type == Library.SPARK:
        to_test = to_test[:1] + to_test[2:]

    features = []
    for test in to_test:
        features.append(
            Feature(
                [
                    Feature(es["log"].ww["value"]),
                    Feature(es["log"].ww["value_2"]),
                ],
                primitive=test[0],
            ),
        )

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2, 3],
    )
    df = to_pandas(df, index="id", sort_index=True)

    for i, test in enumerate(to_test[:-1]):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]
    i, test = -1, to_test[-1]
    v = df[features[i].get_name()].tolist()
    assert np.isnan(v[0])
    assert v[1:] == test[1][1:]


def test_arithmetic_of_direct(es):
    rating = Feature(es["products"].ww["rating"])
    log_rating = Feature(rating, "log")
    customer_age = Feature(es["customers"].ww["age"])
    session_age = Feature(customer_age, "sessions")
    log_age = Feature(session_age, "log")

    to_test = [
        (AddNumeric, [38, 37, 37.5, 37.5]),
        (SubtractNumeric, [28, 29, 28.5, 28.5]),
        (MultiplyNumeric, [165, 132, 148.5, 148.5]),
        (DivideNumeric, [6.6, 8.25, 22.0 / 3, 22.0 / 3]),
    ]
    if es.dataframe_type == Library.SPARK:
        to_test = to_test[:1] + to_test[2:]

    features = []
    for test in to_test:
        features.append(Feature([log_age, log_rating], primitive=test[0]))

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 3, 5, 7],
    )
    df = to_pandas(df, index="id", sort_index=True)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


# Spark EntitySets do not support boolean multiplication
@pytest.fixture(params=["pd_boolean_mult_es", "dask_boolean_mult_es"])
def boolean_mult_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_boolean_mult_es():
    es = EntitySet()
    df = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "bool": pd.Series([True, False, True]),
            "numeric": [2, 3, np.nan],
        },
    )

    es.add_dataframe(
        dataframe_name="test",
        dataframe=df,
        index="index",
        logical_types={"numeric": Double},
    )

    return es


@pytest.fixture
def dask_boolean_mult_es(pd_boolean_mult_es):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    dataframes = {}
    for df in pd_boolean_mult_es.dataframes:
        dataframes[df.ww.name] = (
            dd.from_pandas(df, npartitions=2),
            df.ww.index,
            None,
            df.ww.logical_types,
        )

    return EntitySet(id=pd_boolean_mult_es.id, dataframes=dataframes)


def test_boolean_multiply(boolean_mult_es):
    es = boolean_mult_es
    to_test = [
        ("numeric", "numeric"),
        ("numeric", "bool"),
        ("bool", "numeric"),
        ("bool", "bool"),
    ]
    features = []
    for row in to_test:
        features.append(Feature(es["test"].ww[row[0]]) * Feature(es["test"].ww[row[1]]))

    fm = to_pandas(calculate_feature_matrix(entityset=es, features=features))

    df = to_pandas(es["test"])

    for row in to_test:
        col_name = "{} * {}".format(row[0], row[1])
        if row[0] == "bool" and row[1] == "bool":
            assert fm[col_name].equals((df[row[0]] & df[row[1]]).astype("boolean"))
        else:
            assert fm[col_name].equals(df[row[0]] * df[row[1]])


# TODO: rework test to be Dask and Spark compatible
def test_arithmetic_of_transform(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Test uses Diff which is not supported in Dask or Spark")
    diff1 = Feature([Feature(es["log"].ww["value"])], primitive=Diff)
    diff2 = Feature([Feature(es["log"].ww["value_2"])], primitive=Diff)

    to_test = [
        (AddNumeric, [np.nan, 7.0, -7.0, 10.0]),
        (SubtractNumeric, [np.nan, 3.0, -3.0, 4.0]),
        (MultiplyNumeric, [np.nan, 10.0, 10.0, 21.0]),
        (DivideNumeric, [np.nan, 2.5, 2.5, 2.3333333333333335]),
    ]

    features = []
    for test in to_test:
        features.append(Feature([diff1, diff2], primitive=test[0]()))

    feature_set = FeatureSet(features)
    calculator = FeatureSetCalculator(es, feature_set=feature_set)
    df = calculator.run(np.array([0, 2, 12, 13]))
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert np.isnan(v.pop(0))
        assert np.isnan(test[1].pop(0))
        assert v == test[1]


def test_not_feature(es):
    not_feat = Feature(es["customers"].ww["loves_ice_cream"], primitive=Not)
    features = [not_feat]
    df = to_pandas(
        calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1]),
    )
    v = df[not_feat.get_name()].values
    assert not v[0]
    assert v[1]


def test_arithmetic_of_agg(es):
    customer_id_feat = Feature(es["customers"].ww["id"])
    store_id_feat = Feature(es["stores"].ww["id"])
    count_customer = Feature(
        customer_id_feat,
        parent_dataframe_name="régions",
        primitive=Count,
    )
    count_stores = Feature(
        store_id_feat,
        parent_dataframe_name="régions",
        primitive=Count,
    )
    to_test = [
        (AddNumeric, [6, 2]),
        (SubtractNumeric, [0, -2]),
        (MultiplyNumeric, [9, 0]),
        (DivideNumeric, [1, 0]),
    ]
    # Skip SubtractNumeric for Spark as it's unsupported
    if es.dataframe_type == Library.SPARK:
        to_test = to_test[:1] + to_test[2:]

    features = []
    for test in to_test:
        features.append(Feature([count_customer, count_stores], primitive=test[0]()))

    ids = ["United States", "Mexico"]
    df = calculate_feature_matrix(entityset=es, features=features, instance_ids=ids)
    df = to_pandas(df, index="id", sort_index=True)
    df = df.loc[ids]

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test[1]


def test_latlong(pd_es):
    log_latlong_feat = Feature(pd_es["log"].ww["latlong"])
    latitude = Feature(log_latlong_feat, primitive=Latitude)
    longitude = Feature(log_latlong_feat, primitive=Longitude)
    features = [latitude, longitude]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    latvalues = df[latitude.get_name()].values
    lonvalues = df[longitude.get_name()].values
    assert len(latvalues) == 15
    assert len(lonvalues) == 15
    real_lats = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    real_lons = [0, 2, 4, 6, 8, 0, 1, 2, 3, 0, 0, 2, 0, 3, 6]
    for (
        i,
        v,
    ) in enumerate(real_lats):
        assert v == latvalues[i]
    for (
        i,
        v,
    ) in enumerate(real_lons):
        assert v == lonvalues[i]


def test_latlong_with_nan(pd_es):
    df = pd_es["log"]
    df["latlong"][0] = np.nan
    df["latlong"][1] = (10, np.nan)
    df["latlong"][2] = (np.nan, 4)
    df["latlong"][3] = (np.nan, np.nan)
    pd_es.replace_dataframe(dataframe_name="log", df=df)
    log_latlong_feat = Feature(pd_es["log"].ww["latlong"])
    latitude = Feature(log_latlong_feat, primitive=Latitude)
    longitude = Feature(log_latlong_feat, primitive=Longitude)
    features = [latitude, longitude]
    fm = calculate_feature_matrix(entityset=pd_es, features=features)
    latvalues = fm[latitude.get_name()].values
    lonvalues = fm[longitude.get_name()].values
    assert len(latvalues) == 17
    assert len(lonvalues) == 17
    real_lats = [
        np.nan,
        10,
        np.nan,
        np.nan,
        20,
        0,
        1,
        2,
        3,
        0,
        0,
        5,
        0,
        7,
        14,
        np.nan,
        np.nan,
    ]
    real_lons = [
        np.nan,
        np.nan,
        4,
        np.nan,
        8,
        0,
        1,
        2,
        3,
        0,
        0,
        2,
        0,
        3,
        6,
        np.nan,
        np.nan,
    ]
    assert np.allclose(latvalues, real_lats, atol=0.0001, equal_nan=True)
    assert np.allclose(lonvalues, real_lons, atol=0.0001, equal_nan=True)


def test_haversine(pd_es):
    log_latlong_feat = Feature(pd_es["log"].ww["latlong"])
    log_latlong_feat2 = Feature(pd_es["log"].ww["latlong2"])
    haversine = Feature([log_latlong_feat, log_latlong_feat2], primitive=Haversine)
    features = [haversine]

    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    values = df[haversine.get_name()].values
    real = [
        0,
        525.318462,
        1045.32190304,
        1554.56176802,
        2047.3294327,
        0,
        138.16578931,
        276.20524822,
        413.99185444,
        0,
        0,
        525.318462,
        0,
        741.57941183,
        1467.52760175,
    ]
    assert len(values) == 15
    assert np.allclose(values, real, atol=0.0001)

    haversine = Feature(
        [log_latlong_feat, log_latlong_feat2],
        primitive=Haversine(unit="kilometers"),
    )
    features = [haversine]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    values = df[haversine.get_name()].values
    real_km = [
        0,
        845.41812212,
        1682.2825471,
        2501.82467535,
        3294.85736668,
        0,
        222.35628593,
        444.50926278,
        666.25531268,
        0,
        0,
        845.41812212,
        0,
        1193.45638714,
        2361.75676089,
    ]
    assert len(values) == 15
    assert np.allclose(values, real_km, atol=0.0001)
    error_text = "Invalid unit inches provided. Must be one of"
    with pytest.raises(ValueError, match=error_text):
        Haversine(unit="inches")


def test_haversine_with_nan(pd_es):
    # Check some `nan` values
    df = pd_es["log"]
    df["latlong"][0] = np.nan
    df["latlong"][1] = (10, np.nan)
    pd_es.replace_dataframe(dataframe_name="log", df=df)
    log_latlong_feat = Feature(pd_es["log"].ww["latlong"])
    log_latlong_feat2 = Feature(pd_es["log"].ww["latlong2"])
    haversine = Feature([log_latlong_feat, log_latlong_feat2], primitive=Haversine)
    features = [haversine]

    df = calculate_feature_matrix(entityset=pd_es, features=features)
    values = df[haversine.get_name()].values
    real = [
        np.nan,
        np.nan,
        1045.32190304,
        1554.56176802,
        2047.3294327,
        0,
        138.16578931,
        276.20524822,
        413.99185444,
        0,
        0,
        525.318462,
        0,
        741.57941183,
        1467.52760175,
        np.nan,
        np.nan,
    ]

    assert np.allclose(values, real, atol=0.0001, equal_nan=True)

    # Check all `nan` values
    df = pd_es["log"]
    df["latlong2"] = np.nan
    pd_es.replace_dataframe(dataframe_name="log", df=df)
    log_latlong_feat = Feature(pd_es["log"].ww["latlong"])
    log_latlong_feat2 = Feature(pd_es["log"].ww["latlong2"])
    haversine = Feature([log_latlong_feat, log_latlong_feat2], primitive=Haversine)
    features = [haversine]

    df = calculate_feature_matrix(entityset=pd_es, features=features)
    values = df[haversine.get_name()].values
    real = [np.nan] * pd_es["log"].shape[0]

    assert np.allclose(values, real, atol=0.0001, equal_nan=True)


def test_text_primitives(es):
    words = Feature(es["log"].ww["comments"], primitive=NumWords)
    chars = Feature(es["log"].ww["comments"], primitive=NumCharacters)

    features = [words, chars]

    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(15),
        ),
        index="id",
        sort_index=True,
    )

    word_counts = [532, 3, 3, 653, 1306, 1305, 174, 173, 79, 246, 1253, 3, 3, 3, 3]
    char_counts = [
        3392,
        10,
        10,
        4116,
        7961,
        7580,
        992,
        957,
        437,
        1325,
        6322,
        10,
        10,
        10,
        10,
    ]
    word_values = df[words.get_name()].values
    char_values = df[chars.get_name()].values
    assert len(word_values) == 15
    for i, v in enumerate(word_values):
        assert v == word_counts[i]
    for i, v in enumerate(char_values):
        assert v == char_counts[i]


def test_isin_feat(es):
    isin = Feature(
        es["log"].ww["product_id"],
        primitive=IsIn(list_of_outputs=["toothpaste", "coke zero"]),
    )
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].tolist()
    assert true == v


def test_isin_feat_other_syntax(es):
    isin = Feature(es["log"].ww["product_id"]).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].tolist()
    assert true == v


def test_isin_feat_other_syntax_int(es):
    isin = Feature(es["log"].ww["value"]).isin([5, 10])
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].tolist()
    assert true == v


def test_isin_feat_custom(es):
    class CustomIsIn(TransformPrimitive):
        name = "is_in"
        input_types = [ColumnSchema()]
        return_type = ColumnSchema(logical_type=Boolean)

        def __init__(self, list_of_outputs=None):
            self.list_of_outputs = list_of_outputs

        def get_function(self):
            def pd_is_in(array):
                return array.isin(self.list_of_outputs)

            return pd_is_in

    isin = Feature(
        es["log"].ww["product_id"],
        primitive=CustomIsIn(list_of_outputs=["toothpaste", "coke zero"]),
    )
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].tolist()
    assert true == v

    isin = Feature(es["log"].ww["product_id"]).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].tolist()
    assert true == v

    isin = Feature(es["log"].ww["value"]).isin([5, 10])
    features = [isin]
    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=range(8),
        ),
        index="id",
        sort_index=True,
    )
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].tolist()
    assert true == v


def test_isnull_feat(pd_es):
    value = Feature(pd_es["log"].ww["value"])
    diff = Feature(
        value,
        groupby=Feature(pd_es["log"].ww["session_id"]),
        primitive=Diff,
    )
    isnull = Feature(diff, primitive=IsNull)
    features = [isnull]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )

    correct_vals = [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    values = df[isnull.get_name()].tolist()
    assert correct_vals == values


def test_percentile(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    feature_set = FeatureSet([p])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array(range(10, 17)))
    true = pd_es["log"][v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_dependent_percentile(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    p2 = Feature(p - 1, primitive=Percentile)
    feature_set = FeatureSet([p, p2])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array(range(10, 17)))
    true = pd_es["log"][v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_agg_percentile(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    agg = Feature(p, parent_dataframe_name="sessions", primitive=Sum)
    feature_set = FeatureSet([agg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))
    log_vals = pd_es["log"][[v.get_name(), "session_id"]]
    log_vals["percentile"] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby("session_id")["percentile"].sum()[[0, 1]]
    for t, a in zip(true_p.values, df[agg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg_percentile(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    agg = Feature(p, parent_dataframe_name="sessions", primitive=Sum)
    pagg = Feature(agg, primitive=Percentile)
    feature_set = FeatureSet([pagg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es["log"][[v.get_name(), "session_id"]]
    log_vals["percentile"] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby("session_id")["percentile"].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    agg = Feature(v, parent_dataframe_name="sessions", primitive=Sum)
    pagg = Feature(agg, primitive=Percentile)
    feature_set = FeatureSet([pagg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es["log"][[v.get_name(), "session_id"]]
    true_p = log_vals.groupby("session_id")[v.get_name()].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_percentile(pd_es):
    v = Feature(pd_es["customers"].ww["age"])
    p = Feature(v, primitive=Percentile)
    d = Feature(p, "sessions")
    feature_set = FeatureSet([d])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    cust_vals = pd_es["customers"][[v.get_name()]]
    cust_vals["percentile"] = cust_vals[v.get_name()].rank(pct=True)
    true_p = cust_vals["percentile"].loc[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_agg_percentile(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    agg = Feature(p, parent_dataframe_name="customers", primitive=Sum)
    d = Feature(agg, "sessions")
    feature_set = FeatureSet([d])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es["log"][[v.get_name(), "session_id"]]
    log_vals["percentile"] = log_vals[v.get_name()].rank(pct=True)
    log_vals["customer_id"] = [0] * 10 + [1] * 5 + [2] * 2
    true_p = log_vals.groupby("customer_id")["percentile"].sum().fillna(0)
    true_p = true_p[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or round(t, 3) == round(a, 3)


def test_percentile_with_cutoff(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    p = Feature(v, primitive=Percentile)
    feature_set = FeatureSet([p])
    calculator = FeatureSetCalculator(
        pd_es,
        feature_set,
        pd.Timestamp("2011/04/09 10:30:13"),
    )
    df = calculator.run(np.array([2]))
    assert df[p.get_name()].tolist()[0] == 1.0


def test_two_kinds_of_dependents(pd_es):
    v = Feature(pd_es["log"].ww["value"])
    product = Feature(pd_es["log"].ww["product_id"])
    agg = Feature(
        v,
        parent_dataframe_name="customers",
        where=product == "coke zero",
        primitive=Sum,
    )
    p = Feature(agg, primitive=Percentile)
    g = Feature(agg, primitive=Absolute)
    agg2 = Feature(
        v,
        parent_dataframe_name="sessions",
        where=product == "coke zero",
        primitive=Sum,
    )
    agg3 = Feature(agg2, parent_dataframe_name="customers", primitive=Sum)
    feature_set = FeatureSet([p, g, agg3])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))
    assert df[p.get_name()].tolist() == [2.0 / 3, 1.0]
    assert df[g.get_name()].tolist() == [15, 26]


def test_get_filepath(es):
    class Mod4(TransformPrimitive):
        """Return base feature modulo 4"""

        name = "mod4"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

        def get_function(self):
            filepath = self.get_filepath("featuretools_unit_test_example.csv")
            reference = pd.read_csv(filepath, header=None).squeeze("columns")

            def map_to_word(x):
                def _map(x):
                    if pd.isnull(x):
                        return x
                    return reference[int(x) % 4]

                return x.apply(_map)

            return map_to_word

    feat = Feature(es["log"].ww["value"], primitive=Mod4)
    df = calculate_feature_matrix(features=[feat], entityset=es, instance_ids=range(17))
    df = to_pandas(df, index="id")
    assert pd.isnull(df["MOD4(value)"][15])
    assert df["MOD4(value)"][0] == 0
    assert df["MOD4(value)"][14] == 2

    fm, fl = dfs(
        entityset=es,
        target_dataframe_name="log",
        agg_primitives=[],
        trans_primitives=[Mod4],
    )
    fm = to_pandas(fm, index="id")
    assert fm["MOD4(value)"][0] == 0
    assert fm["MOD4(value)"][14] == 2
    assert pd.isnull(fm["MOD4(value)"][15])


def test_override_multi_feature_names(pd_es):
    def gen_custom_names(primitive, base_feature_names):
        return [
            "Above18(%s)" % base_feature_names,
            "Above21(%s)" % base_feature_names,
            "Above65(%s)" % base_feature_names,
        ]

    class IsGreater(TransformPrimitive):
        name = "is_greater"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

        def get_function(self):
            def is_greater(x):
                return x > 18, x > 21, x > 65

            return is_greater

        def generate_names(primitive, base_feature_names):
            return gen_custom_names(primitive, base_feature_names)

    fm, features = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        instance_ids=[0, 1, 2],
        agg_primitives=[],
        trans_primitives=[IsGreater],
    )

    expected_names = gen_custom_names(IsGreater, ["age"])

    for name in expected_names:
        assert name in fm.columns


def test_time_since_primitive_matches_all_datetime_types(es):
    if es.dataframe_type == Library.SPARK:
        pytest.xfail("TimeSince transform primitive is incompatible with Spark")
    fm, fl = dfs(
        target_dataframe_name="customers",
        entityset=es,
        trans_primitives=[TimeSince],
        agg_primitives=[],
        max_depth=1,
    )

    customers_datetime_cols = [
        id
        for id, t in es["customers"].ww.logical_types.items()
        if isinstance(t, Datetime)
    ]
    expected_names = [f"TIME_SINCE({v})" for v in customers_datetime_cols]

    for name in expected_names:
        assert name in fm.columns


def test_cfm_with_numeric_lag_and_non_nullable_column(pd_es):
    # fill nans so we can use non nullable numeric logical type in the EntitySet
    new_log = pd_es["log"].copy()
    new_log["value"] = new_log["value"].fillna(0)
    new_log.ww.init(
        logical_types={"value": "Integer", "product_id": "Categorical"},
        index="id",
        time_index="datetime",
        name="new_log",
    )
    pd_es.add_dataframe(new_log)
    rels = [
        ("sessions", "id", "new_log", "session_id"),
        ("products", "id", "new_log", "product_id"),
    ]
    pd_es = pd_es.add_relationships(rels)

    assert isinstance(pd_es["new_log"].ww.logical_types["value"], Integer)

    periods = 5
    lag_primitive = NumericLag(periods=periods)
    cutoff_times = pd_es["new_log"][["id", "datetime"]]
    fm, _ = dfs(
        target_dataframe_name="new_log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[lag_primitive],
        cutoff_time=cutoff_times,
    )
    assert fm["NUMERIC_LAG(datetime, value, periods=5)"].head(periods).isnull().all()
    assert fm["NUMERIC_LAG(datetime, value, periods=5)"].isnull().sum() == periods

    assert "NUMERIC_LAG(datetime, value_2, periods=5)" in fm.columns

    assert "NUMERIC_LAG(datetime, products.rating, periods=5)" in fm.columns
    assert (
        fm["NUMERIC_LAG(datetime, products.rating, periods=5)"]
        .head(periods)
        .isnull()
        .all()
    )


def test_cfm_with_lag_and_non_nullable_columns(pd_es):
    # fill nans so we can use non nullable numeric logical type in the EntitySet
    new_log = pd_es["log"].copy()
    new_log["value"] = new_log["value"].fillna(0)
    new_log["value_double"] = new_log["value"]
    new_log["purchased_with_nulls"] = new_log["purchased"]
    new_log["purchased_with_nulls"][0:4] = None
    new_log.ww.init(
        logical_types={
            "value": "Integer",
            "value_2": "IntegerNullable",
            "product_id": "Categorical",
            "value_double": "Double",
            "purchased_with_nulls": "BooleanNullable",
        },
        index="id",
        time_index="datetime",
        name="new_log",
    )
    pd_es.add_dataframe(new_log)
    rels = [
        ("sessions", "id", "new_log", "session_id"),
        ("products", "id", "new_log", "product_id"),
    ]
    pd_es = pd_es.add_relationships(rels)

    assert isinstance(pd_es["new_log"].ww.logical_types["value"], Integer)

    periods = 5
    lag_primitive = Lag(periods=periods)
    cutoff_times = pd_es["new_log"][["id", "datetime"]]
    fm, _ = dfs(
        target_dataframe_name="new_log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[lag_primitive],
        cutoff_time=cutoff_times,
    )
    # Integer
    assert fm["LAG(value, datetime, periods=5)"].head(periods).isnull().all()
    assert fm["LAG(value, datetime, periods=5)"].isnull().sum() == periods
    assert isinstance(
        fm.ww.schema.logical_types["LAG(value, datetime, periods=5)"],
        IntegerNullable,
    )

    # IntegerNullable
    assert "LAG(value_2, datetime, periods=5)" in fm.columns
    assert fm["LAG(value_2, datetime, periods=5)"].head(periods).isnull().all()
    assert isinstance(
        fm.ww.schema.logical_types["LAG(value_2, datetime, periods=5)"],
        IntegerNullable,
    )

    # Categorical
    assert "LAG(product_id, datetime, periods=5)" in fm.columns
    assert fm["LAG(product_id, datetime, periods=5)"].head(periods).isnull().all()
    assert isinstance(
        fm.ww.schema.logical_types["LAG(product_id, datetime, periods=5)"],
        Categorical,
    )

    # Double
    assert "LAG(value_double, datetime, periods=5)" in fm.columns
    assert fm["LAG(value_double, datetime, periods=5)"].head(periods).isnull().all()
    assert isinstance(
        fm.ww.schema.logical_types["LAG(value_double, datetime, periods=5)"],
        Double,
    )

    # Boolean
    assert "LAG(purchased, datetime, periods=5)" in fm.columns
    assert fm["LAG(purchased, datetime, periods=5)"].head(periods).isnull().all()
    assert isinstance(
        fm.ww.schema.logical_types["LAG(purchased, datetime, periods=5)"],
        BooleanNullable,
    )

    # BooleanNullable
    assert "LAG(purchased_with_nulls, datetime, periods=5)" in fm.columns
    assert (
        fm["LAG(purchased_with_nulls, datetime, periods=5)"]
        .head(periods)
        .isnull()
        .all()
    )
    assert isinstance(
        fm.ww.schema.logical_types["LAG(purchased_with_nulls, datetime, periods=5)"],
        BooleanNullable,
    )


def test_comparisons_with_ordinal_valid_inputs_that_dont_work_but_should(pd_es):
    # TODO: Remvoe this test once the correct behavior is implemented in CFM
    # The following test covers a scenario where an intermediate feature doesn't have the correct type
    # because Woodwork has not yet been initialized. This calculation should work and return valid True/False
    # values. This should be fixed in a future PR, but until a fix is implemented null values are returned to
    # prevent calculate_feature_matrix from raising an Error when calculating features generated by DFS.

    priority_level = Feature(pd_es["log"].ww["priority_level"])
    first_priority = AggregationFeature(
        priority_level,
        parent_dataframe_name="customers",
        primitive=First,
    )
    engagement = Feature(pd_es["customers"].ww["engagement_level"])
    invalid_but_should_be_valid = [
        TransformFeature([engagement, first_priority], primitive=LessThan),
        TransformFeature([engagement, first_priority], primitive=LessThanEqualTo),
        TransformFeature([engagement, first_priority], primitive=GreaterThan),
        TransformFeature([engagement, first_priority], primitive=GreaterThanEqualTo),
    ]
    fm = calculate_feature_matrix(
        entityset=pd_es,
        features=invalid_but_should_be_valid,
    )

    feature_cols = [f.get_name() for f in invalid_but_should_be_valid]
    fm = to_pandas(fm)
    for col in feature_cols:
        assert fm[col].isnull().all()


def test_multiply_numeric_boolean():
    test_cases = [
        {"val": 100, "mask": True, "expected": 100},
        {"val": 100, "mask": False, "expected": 0},
        {"val": 0, "mask": False, "expected": 0},
        {"val": 100, "mask": pd.NA, "expected": pd.NA},
        {"val": pd.NA, "mask": pd.NA, "expected": pd.NA},
        {"val": pd.NA, "mask": True, "expected": pd.NA},
        {"val": pd.NA, "mask": False, "expected": pd.NA},
    ]

    multiply_numeric_boolean = MultiplyNumericBoolean()
    for input in test_cases:
        vals = pd.Series(input["val"]).astype("Int64")
        mask = pd.Series(input["mask"])
        actual = multiply_numeric_boolean(vals, mask).tolist()[0]
        expected = input["expected"]
        if pd.isnull(expected):
            assert pd.isnull(actual)
        else:
            assert actual == input["expected"]


def test_multiply_numeric_boolean_multiple_dtypes_no_nulls():
    # Test without null values
    vals = pd.Series([1, 2, 3])
    bools = pd.Series([True, False, True])
    multiply_numeric_boolean = MultiplyNumericBoolean()
    numeric_dtypes = ["float64", "int64", "Int64"]
    boolean_dtypes = ["bool", "boolean"]

    for numeric_dtype in numeric_dtypes:
        for boolean_dtype in boolean_dtypes:
            actual = multiply_numeric_boolean(
                vals.astype(numeric_dtype),
                bools.astype(boolean_dtype),
            )
            expected = pd.Series([1, 0, 3])
            pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_multiply_numeric_boolean_multiple_dtypes_with_nulls():
    # Test with null values
    vals = pd.Series([np.nan, 2, 3])
    bools = pd.Series([True, False, pd.NA], dtype="boolean")
    multiply_numeric_boolean = MultiplyNumericBoolean()
    numeric_dtypes = ["float64", "Int64"]

    for numeric_dtype in numeric_dtypes:
        actual = multiply_numeric_boolean(vals.astype(numeric_dtype), bools)
        expected = pd.Series([np.nan, 0, np.nan])
        pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_feature_multiplication(es):
    numeric_ft = Feature(es["customers"].ww["age"])
    boolean_ft = Feature(es["customers"].ww["loves_ice_cream"])

    mult_numeric = numeric_ft * numeric_ft
    mult_boolean = boolean_ft * boolean_ft
    mult_numeric_boolean = numeric_ft * boolean_ft
    mult_numeric_boolean2 = boolean_ft * numeric_ft

    assert issubclass(type(mult_numeric.primitive), MultiplyNumeric)
    assert issubclass(type(mult_boolean.primitive), MultiplyBoolean)
    assert issubclass(type(mult_numeric_boolean.primitive), MultiplyNumericBoolean)
    assert issubclass(type(mult_numeric_boolean2.primitive), MultiplyNumericBoolean)

    # Test with nullable types
    es["customers"].ww.set_types(
        logical_types={"age": "IntegerNullable", "loves_ice_cream": "BooleanNullable"},
    )
    numeric_ft = Feature(es["customers"].ww["age"])
    boolean_ft = Feature(es["customers"].ww["loves_ice_cream"])
    mult_numeric = numeric_ft * numeric_ft
    mult_boolean = boolean_ft * boolean_ft
    mult_numeric_boolean = numeric_ft * boolean_ft
    mult_numeric_boolean2 = boolean_ft * numeric_ft

    assert issubclass(type(mult_numeric.primitive), MultiplyNumeric)
    assert issubclass(type(mult_boolean.primitive), MultiplyBoolean)
    assert issubclass(type(mult_numeric_boolean.primitive), MultiplyNumericBoolean)
    assert issubclass(type(mult_numeric_boolean2.primitive), MultiplyNumericBoolean)
