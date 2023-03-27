import hashlib
import json
from typing import List, Type

import pytest
from woodwork.logical_types import Double

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import FeatureBase
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.feature_discovery.feature_discovery import my_dfs
from featuretools.feature_discovery.type_defs import (
    Feature,
    convert_feature_to_featurebase,
    convert_featurebase_to_feature,
)
from featuretools.primitives import AddNumeric, DivideNumeric
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.standard.transform.numeric.absolute import Absolute
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_feature_type_equality():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)

    # Add Numeric is Commutative, so should all be equal
    f3 = Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    f4 = Feature(
        name="Column 10",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    f5 = Feature(
        name="Column 20",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f3 == f4 == f5

    # Divide Numeric is not Commutative, so should not be equal
    f6 = Feature(
        name="Column 1",
        primitive=DivideNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    f7 = Feature(
        name="Column 1",
        primitive=DivideNumeric,
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
            primitive=AddNumeric,
            logical_type=Double,
        )


def test_feature_to_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f = Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    expected = {
        "name": "Column 1",
        "logical_type": "Double",
        "tags": [],
        "primitive": "AddNumeric",
        "base_features": [
            {
                "name": "f1",
                "logical_type": "Double",
                "tags": [],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "3f524cdc07a11d7c6220bdb049fe8dd41b27483c96cc59b581e022d547290d69",
            },
            {
                "name": "f2",
                "logical_type": "Double",
                "tags": [],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "e4ab4e3b1493d5a997b4e51cdefbaa10570ef3ea9432bd72e7b6a89654ceb7f6",
            },
        ],
        "df_id": None,
        "id": "b999e3fdddba9cfca2d63fe4030332210578f07698f21e82d3c91c094b1862cf",
    }

    actual = f.to_dict()
    json_str = json.dumps(actual)
    assert actual == expected
    assert json.dumps(expected) == json_str


def test_feature_from_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f_orig = Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    input_dict = f_orig.to_dict()
    f_from_dict = Feature.from_dict(input_dict)
    assert f_orig == f_from_dict


def test_convert_featurebase_to_feature():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    fdefs = dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=[AddNumeric],
        features_only=True,
    )
    assert isinstance(fdefs, list)
    assert isinstance(fdefs[0], FeatureBase)

    converted_features = set([convert_featurebase_to_feature(x) for x in fdefs])

    f1 = Feature("f_1", Double)
    f2 = Feature("f_2", Double)
    f3 = Feature(
        name="f_1 + f_2",
        tags={"numeric"},
        primitive=AddNumeric,
        base_features=[f1, f2],
    )

    orig_features = set([f1, f2, f3])

    assert len(orig_features.symmetric_difference(converted_features)) == 0


def hash_old_feature(feature: FeatureBase):
    feature_dict = feature.to_dictionary()
    # "type": type(self).__name__,
    #         "dependencies": [dep.unique_name() for dep in self.get_dependencies()],
    #         "arguments": self.get_arguments(),

    hash_msg = hashlib.sha256()

    hash_msg.update(feature_dict["type"])

    for dep in feature_dict["dep"]:
        hash_msg.update(dep)

    # if primitive:
    #     primitive_name = primitive.name
    #     assert isinstance(primitive_name, str)
    #     commutative = primitive.commutative
    #     hash_msg.update(primitive_name.encode("utf-8"))

    #     assert (
    #         len(base_features) > 0
    #     ), "there must be base features if give a primitive"
    #     base_columns = base_features
    #     if commutative:
    #         base_features.sort()

    #     for c in base_columns:
    #         hash_msg.update(c.id.encode("utf-8"))

    # else:
    #     assert name
    #     hash_msg.update(name.encode("utf-8"))

    return hash_msg.hexdigest()


def test_convert_feature_to_feature_base():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    primitives: List[Type[PrimitiveBase]] = [Absolute]

    features_old = dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=primitives,
        features_only=True,
    )
    features_new = my_dfs(df.ww.schema, primitives)

    features_new = [x for x in features_new if "index" not in x.tags]

    converted_features = [convert_feature_to_featurebase(x, es) for x in features_new]

    f1 = set(FeaturesSerializer(features_old).to_dict()["feature_list"])

    f2 = set(FeaturesSerializer(converted_features).to_dict()["feature_list"])

    assert f1 == f2
