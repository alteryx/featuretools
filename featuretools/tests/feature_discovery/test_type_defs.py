from woodwork.logical_types import Double

from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import AddNumeric


def test_feature_type():

    Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
    )
