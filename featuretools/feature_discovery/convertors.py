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


def convert_featurebase_list_to_feature_list(
    featurebase_list: List[FeatureBase],
) -> List[LiteFeature]:
    """
    Convert a List of FeatureBase objects to a list LiteFeature objects

    Args:
        featurebase_list (List[FeatureBase]):

    Returns:
       LiteFeatures (List[LiteFeature]) - converted LiteFeature objects
    """

    def rfunc(fb: FeatureBase) -> List[LiteFeature]:
        base_features = [
            feature
            for feature_list in [rfunc(x) for x in fb.base_features]
            for feature in feature_list
        ]
        col_schema = fb.column_schema

        logical_type = col_schema.logical_type
        if logical_type is not None:
            assert issubclass(type(logical_type), LogicalType)
            logical_type = type(logical_type)

        tags = col_schema.semantic_tags

        if isinstance(fb, IdentityFeature):
            primitive = None
        else:
            primitive = fb.primitive
            assert isinstance(primitive, PrimitiveBase)

        if fb.number_output_features > 1:
            features: List[LiteFeature] = []

            for idx, name in enumerate(fb.get_feature_names()):
                f = LiteFeature(
                    name=name,
                    logical_type=logical_type,
                    tags=tags,
                    primitive=primitive,
                    base_features=base_features,
                    # TODO: use when working with multi-table
                    df_id=None,
                    idx=idx,
                )
                features.append(f)

            for feature in features:
                related_features = [f for f in features if f.id != feature.id]
                feature.related_features = set(related_features)

            return features

        return [
            LiteFeature(
                name=fb.get_name(),
                logical_type=logical_type,
                tags=tags,
                primitive=primitive,
                base_features=base_features,
                # TODO: use when working with multi-table
                df_id=None,
            ),
        ]

    return [
        feature
        for feature_list in [rfunc(fb) for fb in featurebase_list]
        for feature in feature_list
    ]


def _feature_to_transform_feature(
    feature: LiteFeature,
    base_features: List[FeatureBase],
) -> FeatureBase:
    """
    Transform LiteFeature into FeatureBase object. Handles the Multi-output
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
    if feature.is_multioutput():
        sorted_features = sorted(
            [f for f in feature.related_features] + [feature],
            key=lambda x: x.idx,
        )
        names = [x.name for x in sorted_features]

        fb = fb.rename(feature.non_indexed_name)
        fb.set_feature_names(names)
    else:
        fb = fb.rename(feature.name)

    return fb


def _convert_feature_to_featurebase(
    feature: LiteFeature,
    dataframe: pd.DataFrame,
    cache: FeatureCache,
) -> FeatureBase:
    """
    Recursively transforms a LiteFeature object into a Featurebase object

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

        fb = _feature_to_transform_feature(feature, base_features)
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

        fb = _convert_feature_to_featurebase(
            feature=feature,
            dataframe=dataframe,
            cache=feature_cache,
        )
        converted_features.append(fb)

    return converted_features
