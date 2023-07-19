import json
from unittest.mock import patch

import pytest
from woodwork.logical_types import Boolean, Double

from featuretools.feature_discovery.feature_discovery import (
    generate_features_from_primitives,
    schema_to_features,
)
from featuretools.feature_discovery.FeatureCollection import FeatureCollection
from featuretools.feature_discovery.LiteFeature import LiteFeature
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    DivideNumeric,
    Lag,
    MultiplyNumeric,
)
from featuretools.tests.feature_discovery.test_feature_discovery import (
    MultiOutputPrimitiveForTest,
)
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_feature_type_equality():
    f1 = LiteFeature("f1", Double)
    f2 = LiteFeature("f2", Double)

    # Add Numeric is Commutative, so should all be equal
    f3 = LiteFeature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f4 = LiteFeature(
        name="Column 10",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f5 = LiteFeature(
        name="Column 20",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f3 == f4 == f5

    # Divide Numeric is not Commutative, so should not be equal
    f6 = LiteFeature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f7 = LiteFeature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f6 != f7


def test_feature_type_assertions():
    with pytest.raises(
        ValueError,
        match="there must be base features if given a primitive",
    ):
        LiteFeature(
            name="Column 1",
            primitive=AddNumeric(),
            logical_type=Double,
        )


@patch.object(LiteFeature, "_generate_hash", lambda x: x.name)
@patch(
    "featuretools.feature_discovery.LiteFeature.hash_primitive",
    lambda x: (x.name, None),
)
def test_feature_to_dict():
    f1 = LiteFeature("f1", Double)
    f2 = LiteFeature("f2", Double)
    f = LiteFeature(
        name="Column 1",
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )

    expected = {
        "name": "Column 1",
        "logical_type": None,
        "tags": ["numeric"],
        "primitive": "add_numeric",
        "base_features": ["f1", "f2"],
        "df_id": None,
        "id": "Column 1",
        "related_features": [],
        "idx": 0,
    }

    actual = f.to_dict()
    json_str = json.dumps(actual)
    assert actual == expected
    assert json.dumps(expected) == json_str


def test_feature_hash():
    bf1 = LiteFeature("bf", Double)
    bf2 = LiteFeature("bf", Double, df_id="df")

    p1 = Lag(periods=1)
    p2 = Lag(periods=2)
    f1 = LiteFeature(
        primitive=p1,
        logical_type=Double,
        base_features=[bf1],
    )

    f2 = LiteFeature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf1],
    )

    f3 = LiteFeature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf1],
    )

    f4 = LiteFeature(
        primitive=p1,
        logical_type=Double,
        base_features=[bf2],
    )

    # TODO(dreed): ensure ID is parquet and arrow acceptable, length and starting character might be problematic

    assert f1 != f2
    assert f2 == f3
    assert f1 != f4


def test_feature_forced_name():
    bf = LiteFeature("bf", Double)

    p1 = Lag(periods=1)
    f1 = LiteFeature(
        name="target_delay_1",
        primitive=p1,
        logical_type=Double,
        base_features=[bf],
    )
    assert f1.name == "target_delay_1"


@patch.object(LiteFeature, "_generate_hash", lambda x: x.name)
@patch(
    "featuretools.feature_discovery.FeatureCollection.hash_primitive",
    lambda x: (x.name, None),
)
@patch(
    "featuretools.feature_discovery.LiteFeature.hash_primitive",
    lambda x: (x.name, None),
)
def test_feature_collection_to_dict():
    f1 = LiteFeature("f1", Double)
    f2 = LiteFeature("f2", Double)
    f3 = LiteFeature(
        name="Column 1",
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )

    fc = FeatureCollection([f3])

    expected = {
        "primitives": {
            "add_numeric": None,
        },
        "feature_ids": ["Column 1"],
        "all_features": {
            "Column 1": {
                "name": "Column 1",
                "logical_type": None,
                "tags": ["numeric"],
                "primitive": "add_numeric",
                "base_features": ["f1", "f2"],
                "df_id": None,
                "id": "Column 1",
                "related_features": [],
                "idx": 0,
            },
            "f1": {
                "name": "f1",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f1",
                "related_features": [],
                "idx": 0,
            },
            "f2": {
                "name": "f2",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f2",
                "related_features": [],
                "idx": 0,
            },
        },
    }

    actual = fc.to_dict()
    assert actual == expected
    assert json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)


@patch.object(LiteFeature, "_generate_hash", lambda x: x.name)
def test_feature_collection_from_dict():
    f1 = LiteFeature("f1", Double)
    f2 = LiteFeature("f2", Double)
    f3 = LiteFeature(
        name="Column 1",
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )

    expected = FeatureCollection([f3])

    input_dict = {
        "primitives": {
            "009da67f0a1430630c4a419c84aac270ec62337ab20c080e4495272950fd03b3": {
                "type": "AddNumeric",
                "module": "featuretools.primitives.standard.transform.binary.add_numeric",
                "arguments": {},
            },
        },
        "feature_ids": ["Column 1"],
        "all_features": {
            "f2": {
                "name": "f2",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f2",
                "related_features": [],
                "idx": 0,
            },
            "f1": {
                "name": "f1",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f1",
                "related_features": [],
                "idx": 0,
            },
            "Column 1": {
                "name": "Column 1",
                "logical_type": None,
                "tags": ["numeric"],
                "primitive": "009da67f0a1430630c4a419c84aac270ec62337ab20c080e4495272950fd03b3",
                "base_features": ["f1", "f2"],
                "df_id": None,
                "id": "Column 1",
                "related_features": [],
                "idx": 0,
            },
        },
    }

    actual = FeatureCollection.from_dict(input_dict)

    assert actual == expected


@patch.object(LiteFeature, "__lt__", lambda x, y: x.name < y.name)
def test_feature_collection_serialization_roundtrip():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("t_idx", "Datetime", {"time_index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
        ("f_3", "Categorical"),
        ("f_4", "Boolean"),
        ("f_5", "NaturalLanguage"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    origin_features = schema_to_features(df.ww.schema)
    features = generate_features_from_primitives(
        origin_features,
        [Absolute, MultiplyNumeric, MultiOutputPrimitiveForTest],
    )

    features = generate_features_from_primitives(features, [Lag])

    assert set([x.name for x in features]) == set(
        [
            "idx",
            "t_idx",
            "f_1",
            "f_2",
            "f_3",
            "f_4",
            "f_5",
            "ABSOLUTE(f_1)",
            "ABSOLUTE(f_2)",
            "f_1 * f_2",
            "TEST_MO(f_5)[0]",
            "TEST_MO(f_5)[1]",
            "LAG(f_1, t_idx)",
            "LAG(f_2, t_idx)",
            "LAG(f_3, t_idx)",
            "LAG(f_4, t_idx)",
            "LAG(ABSOLUTE(f_1), t_idx)",
            "LAG(ABSOLUTE(f_2), t_idx)",
            "LAG(f_1 * f_2, t_idx)",
            "LAG(TEST_MO(f_5)[1], t_idx)",
            "LAG(TEST_MO(f_5)[0], t_idx)",
        ],
    )
    fc = FeatureCollection(features=features)
    fc_dict = fc.to_dict()

    fc_json = json.dumps(fc_dict)

    fc2_dict = json.loads(fc_json)

    fc2 = FeatureCollection.from_dict(fc2_dict)

    assert fc == fc2
    lsa_features = [x for x in fc2.all_features if x.get_primitive_name() == "test_mo"]
    assert len(lsa_features[0].related_features) == 1


def test_lite_feature_assertions():
    f1 = LiteFeature(name="f1", logical_type=Double)
    f2 = LiteFeature(name="f1", logical_type=Double, df_id="df1")

    assert f1 != f2

    with pytest.raises(
        TypeError,
        match="Name must be given if origin feature",
    ):
        LiteFeature(logical_type=Double)

    with pytest.raises(
        TypeError,
        match="Logical Type must be given if origin feature",
    ):
        LiteFeature(name="f1")

    with pytest.raises(
        ValueError,
        match="primitive input must be of type PrimitiveBase",
    ):
        LiteFeature(name="f3", primitive="AddNumeric", base_features=[f1, f2])

    f = LiteFeature("f4", logical_type=Double)
    with pytest.raises(AttributeError, match="name is immutable"):
        f.name = "new name"

    with pytest.raises(ValueError, match="only used on multioutput features"):
        f.non_indexed_name

    with pytest.raises(AttributeError, match="logical_type is immutable"):
        f.logical_type = Boolean

    with pytest.raises(AttributeError, match="tags is immutable"):
        f.tags = {"other"}

    with pytest.raises(AttributeError, match="primitive is immutable"):
        f.primitive = AddNumeric

    with pytest.raises(AttributeError, match="base_features are immutable"):
        f.base_features = [f1]

    with pytest.raises(AttributeError, match="df_id is immutable"):
        f.df_id = "df_id"

    with pytest.raises(AttributeError, match="id is immutable"):
        f.id = "id"

    with pytest.raises(AttributeError, match="n_output_features is immutable"):
        f.n_output_features = "n_output_features"

    with pytest.raises(AttributeError, match="depth is immutable"):
        f.depth = "depth"

    with pytest.raises(AttributeError, match="idx is immutable"):
        f.idx = "idx"


def test_lite_feature_to_column_schema():
    f1 = LiteFeature(name="f1", logical_type=Double, tags={"index", "numeric"})

    column_schema = f1.column_schema

    assert column_schema.is_numeric
    assert isinstance(column_schema.logical_type, Double)
    assert column_schema.semantic_tags == {"index", "numeric"}

    f2 = LiteFeature(name="f2", primitive=Absolute(), base_features=[f1])

    column_schema = f2.column_schema
    assert column_schema.semantic_tags == {"numeric"}


def test_lite_feature_to_dependent_primitives():
    f1 = LiteFeature(name="f1", logical_type=Double)

    f2 = LiteFeature(name="f2", primitive=Absolute(), base_features=[f1])

    f3 = LiteFeature(name="f3", primitive=AddNumeric(), base_features=[f1, f2])

    f4 = LiteFeature(name="f4", primitive=MultiplyNumeric(), base_features=[f1, f3])

    assert set([x.name for x in f4.dependent_primitives()]) == set(
        ["multiply_numeric", "absolute", "add_numeric"],
    )
