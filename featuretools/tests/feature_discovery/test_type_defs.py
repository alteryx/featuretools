import json
from unittest.mock import patch

import pytest
from woodwork.logical_types import Double

from featuretools.feature_discovery.feature_discovery import my_dfs, schema_to_features
from featuretools.feature_discovery.type_defs import (
    Feature,
    FeatureCollection,
)
from featuretools.primitives import (
    LSA,
    Absolute,
    AddNumeric,
    DivideNumeric,
    Lag,
    MultiplyNumeric,
)
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_feature_type_equality():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)

    # Add Numeric is Commutative, so should all be equal
    f3 = Feature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f4 = Feature(
        name="Column 10",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f5 = Feature(
        name="Column 20",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f3 == f4 == f5

    # Divide Numeric is not Commutative, so should not be equal
    f6 = Feature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f7 = Feature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f6 != f7


def test_feature_type_assertions():
    with pytest.raises(
        AssertionError,
        match="there must be base features if give a primitive",
    ):
        Feature(
            name="Column 1",
            primitive=AddNumeric(),
            logical_type=Double,
        )


# @patch.object(Feature, "_generate_hash", lambda x: x.name)
# def test_feature_to_dict():
#     f1 = Feature("f1", Double)
#     f2 = Feature("f2", Double)
#     f = Feature(
#         name="Column 1",
#         primitive=AddNumeric(),
#         logical_type=Double,
#         base_features=[f1, f2],
#     )

#     expected = {
#         "name": "Column 1",
#         "logical_type": "Double",
#         "tags": ["numeric"],
#         "primitive": {
#             "type": "AddNumeric",
#             "module": "featuretools.primitives.standard.transform.binary.add_numeric",
#             "arguments": {},
#         },
#         "base_features": [
#             {
#                 "name": "f1",
#                 "logical_type": "Double",
#                 "tags": ["numeric"],
#                 "primitive": None,
#                 "base_features": [],
#                 "df_id": None,
#                 "id": "f1",
#                 "related_features"
#             },
#             {
#                 "name": "f2",
#                 "logical_type": "Double",
#                 "tags": ["numeric"],
#                 "primitive": None,
#                 "base_features": [],
#                 "df_id": None,
#                 "id": "f2",
#             },
#         ],
#         "df_id": None,
#         "id": "Column 1",
#     }

#     actual = f.to_dict()
#     json_str = json.dumps(actual)
#     assert actual == expected
#     assert json.dumps(expected) == json_str


def test_feature_from_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f_orig = Feature(
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    input_dict = f_orig.to_dict()
    f_from_dict = Feature.from_dict(input_dict)
    assert f_orig == f_from_dict


def test_feature_hash():
    bf = Feature("bf", Double)

    p1 = Lag(periods=1)
    p2 = Lag(periods=2)
    f1 = Feature(
        primitive=p1,
        logical_type=Double,
        base_features=[bf],
    )

    f2 = Feature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf],
    )

    f3 = Feature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf],
    )

    assert f1 != f2
    assert f2 == f3


def test_feature_forced_name():
    bf = Feature("bf", Double)

    p1 = Lag(periods=1)
    f1 = Feature(
        name="target_delay_1",
        primitive=p1,
        logical_type=Double,
        base_features=[bf],
    )
    assert f1.name == "target_delay_1"


@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_feature_collection_to_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f3 = Feature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    fc = FeatureCollection([f3])

    expected = {
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
                "logical_type": "Double",
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

    actual = fc.to_dict()
    assert actual == expected
    assert json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)


@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_feature_collection_from_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f3 = Feature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
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
                "logical_type": "Double",
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
    fc = my_dfs(origin_features, [Absolute, MultiplyNumeric, LSA])

    fc = my_dfs(fc.all_features, [Lag])

    assert set([x.get_name() for x in fc.all_features]) == set(
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
            "LSA(f_5)[0]",
            "LSA(f_5)[1]",
            "LAG(f_1, t_idx)",
            "LAG(f_2, t_idx)",
            "LAG(f_3, t_idx)",
            "LAG(f_4, t_idx)",
            "LAG(ABSOLUTE(f_1), t_idx)",
            "LAG(ABSOLUTE(f_2), t_idx)",
            "LAG(f_1 * f_2, t_idx)",
            "LAG(LSA(f_5)[1], t_idx)",
            "LAG(LSA(f_5)[0], t_idx)",
        ],
    )

    fc_dict = fc.to_dict()

    fc_json = json.dumps(fc_dict)

    fc2_dict = json.loads(fc_json)

    fc2 = FeatureCollection.from_dict(fc2_dict)

    assert fc == fc2
    lsa_features = [x for x in fc2.all_features if x.get_primitive_name() == "lsa"]
    assert len(lsa_features[0].related_features) == 1
