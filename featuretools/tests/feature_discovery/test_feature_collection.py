import pytest
from woodwork.logical_types import (
    Boolean,
    Double,
    Ordinal,
)

from featuretools.feature_discovery.FeatureCollection import FeatureCollection
from featuretools.feature_discovery.LiteFeature import LiteFeature
from featuretools.primitives import Absolute, AddNumeric


@pytest.mark.parametrize(
    "feature_args, expected",
    [
        (
            ("idx", Double),
            ["ANY", "Double", "Double,numeric", "numeric"],
        ),
        (
            ("idx", Double, {"index"}),
            ["ANY", "Double", "Double,index", "index"],
        ),
        (
            ("idx", Double, {"other"}),
            [
                "ANY",
                "Double",
                "other",
                "numeric",
                "Double,other",
                "Double,numeric",
                "numeric,other",
                "Double,numeric,other",
            ],
        ),
        (
            ("idx", Ordinal, {"other"}),
            [
                "ANY",
                "Ordinal",
                "other",
                "category",
                "Ordinal,other",
                "Ordinal,category",
                "category,other",
                "Ordinal,category,other",
            ],
        ),
        (
            ("idx", Double, {"a", "b", "numeric"}),
            [
                "ANY",
                "Double",
                "a",
                "b",
                "numeric",
                "Double,a",
                "Double,b",
                "Double,numeric",
                "a,b",
                "a,numeric",
                "b,numeric",
                "a,b,numeric",
                "Double,a,b",
                "Double,a,numeric",
                "Double,b,numeric",
                "Double,a,b,numeric",
            ],
        ),
    ],
)
def test_to_keys_method(feature_args, expected):
    feature = LiteFeature(*feature_args)

    keys = FeatureCollection.feature_to_keys(feature)

    assert set(keys) == set(expected)


def test_feature_collection_hashing():
    f1 = LiteFeature(name="f1", logical_type=Double)
    f2 = LiteFeature(name="f2", logical_type=Double, tags={"index"})
    f3 = LiteFeature(name="f3", logical_type=Boolean, tags={"other"})
    f4 = LiteFeature(name="f4", primitive=Absolute(), base_features=[f1])
    f5 = LiteFeature(name="f5", primitive=AddNumeric(), base_features=[f1, f2])

    fc1 = FeatureCollection([f1, f2, f3, f4, f5])
    fc2 = FeatureCollection([f1, f2, f3, f4, f5])

    assert len(set([fc1, fc2])) == 1

    fc1.reindex()
    assert fc1.get_by_logical_type(Double) == set([f1, f2])

    assert fc1.get_by_tag("index") == set([f2])

    assert fc1.get_by_origin_feature(f1) == set([f1, f4, f5])

    assert fc1.get_dependencies_by_origin_name("f1") == set([f1, f4, f5])

    assert fc1.get_dependencies_by_origin_name("null") == set()

    assert fc1.get_by_origin_feature_name("f1") == f1

    assert fc1.get_by_origin_feature_name("null") is None
