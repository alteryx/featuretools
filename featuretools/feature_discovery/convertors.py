from __future__ import annotations

from typing import Dict, List

import pandas as pd
from woodwork.logical_types import LogicalType

from featuretools.feature_base.feature_base import (
    FeatureBase,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_discovery.LiteFeature import LiteFeature
from featuretools.primitives import TransformPrimitive
from featuretools.primitives.base.primitive_base import PrimitiveBase

FeatureCache = Dict[str, FeatureBase]


def convert_featurebase_to_feature(feature: FeatureBase) -> LiteFeature:
    """
    Convert a FeatureBase object to a LiteFeature object

    Args:
        feature (FeatureBase):

    Returns:
       LiteFeature - converted LiteFeature object
    """
    base_features = [convert_featurebase_to_feature(x) for x in feature.base_features]

    name = feature.get_name()
    col_schema = feature.column_schema

    logical_type = col_schema.logical_type
    if logical_type is not None:
        assert issubclass(type(logical_type), LogicalType)
        logical_type = type(logical_type)

    tags = col_schema.semantic_tags

    if isinstance(feature, IdentityFeature):
        primitive = None
    else:
        primitive = feature.primitive
        assert isinstance(primitive, PrimitiveBase)

    return LiteFeature(
        name=name,
        logical_type=logical_type,
        tags=tags,
        primitive=primitive,
        base_features=base_features,
        # TODO: replace this with dataframe name?
        df_id=None,
    )


def to_transform_feature(
    feature: LiteFeature,
    base_features: List[FeatureBase],
) -> FeatureBase:
    """
    Transform feature into FeatureBase object. Handles the Multi-output
    feature in correct way.

    Args:
        feature (LiteFeature)
        base_features (List[FeatureBase])

    Returns:
       FeatureBase
    """
    assert feature.primitive
    assert isinstance(
        feature.primitive,
        TransformPrimitive,
    ), "Only Transform Primitives"

    fb = TransformFeature(base_features, feature.primitive)
    if len(feature.related_features) > 0:
        sorted_features = sorted(
            [f for f in feature.related_features] + [feature],
        )
        names = [x.get_name() for x in sorted_features]

        fb = fb.rename(feature._gen_name)
        fb.set_feature_names(names)
    else:
        fb = fb.rename(feature.get_name())

    return fb


def convert_feature_to_featurebase(
    feature: LiteFeature,
    dataframe: pd.DataFrame,
    cache: FeatureCache = {},
) -> FeatureBase:
    """
    Recursively transforms a feature object into a Featurebase object

    Args:
        feature (LiteFeature)
        base_features (List[FeatureBase])
        cache (FeatureCache) already converted features

    Returns:
       FeatureBase
    """

    def get_base_features(
        feature: LiteFeature,
    ) -> List[FeatureBase]:
        new_base_features: List[FeatureBase] = []
        for bf in feature.base_features:
            fb = rfunc(bf)
            if bf.is_multioutput():
                idx = bf.idx
                # if its multioutput, you can index on the FeatureBase
                new_base_features.append(fb[idx])
            else:
                new_base_features.append(fb)

        return new_base_features

    def rfunc(feature: LiteFeature) -> FeatureBase:
        # if feature has already been converted, return from cache
        if feature.id in cache:
            return cache[feature.id]

        # if depth is 0, we are at an origin feature
        if feature.depth == 0:
            fb = IdentityFeature(dataframe.ww[feature.name])
            cache[feature.id] = fb
            return fb

        base_features = get_base_features(feature)

        fb = to_transform_feature(feature, base_features)
        cache[feature.id] = fb
        return fb

    return rfunc(feature)


def convert_feature_list_to_featurebase_list(
    feature_list: List[LiteFeature],
    dataframe: pd.DataFrame,
) -> List[FeatureBase]:
    """
    Convert a list of LiteFeature objects into a list of FeatureBase objects

    Args:
        feature_list (List[LiteFeature])
        dataframe (pd.DataFrame)

    Returns:
       List[FeatureBase]
    """
    feature_cache: FeatureCache = {}

    converted_features: List[FeatureBase] = []
    for feature in feature_list:
        if feature.is_multioutput():
            related_feature_ids = [f.id for f in feature.related_features]
            if any((x in feature_cache for x in related_feature_ids)):
                # feature base already created for related ids
                continue

        fb = convert_feature_to_featurebase(
            feature=feature,
            dataframe=dataframe,
            cache=feature_cache,
        )
        converted_features.append(fb)

    return converted_features
