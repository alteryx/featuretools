import pytest
from woodwork.logical_types import (
    Double,
    Ordinal,
)

from featuretools.feature_discovery.FeatureCollection import FeatureCollection
from featuretools.feature_discovery.LiteFeature import LiteFeature


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
