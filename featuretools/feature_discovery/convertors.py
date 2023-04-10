from __future__ import annotations

from typing import Dict, List

import pandas as pd
from woodwork.logical_types import LogicalType

from featuretools.feature_base.feature_base import (
    FeatureBase,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import TransformPrimitive
from featuretools.primitives.base.primitive_base import PrimitiveBase

FeatureCache = Dict[str, FeatureBase]


def convert_featurebase_to_feature(feature: FeatureBase) -> Feature:
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

    return Feature(
        name=name,
        logical_type=logical_type,
        tags=tags,
        primitive=primitive,
        base_features=base_features,
        # TODO: replace this with dataframe name?
        df_id=None,
    )


def to_identity_feature(feature: Feature, dataframe: pd.DataFrame) -> FeatureBase:
    fb = IdentityFeature(dataframe.ww[feature.name])
    return fb


def get_base_features(
    feature: Feature,
    feature_cache: FeatureCache,
) -> List[FeatureBase]:
    new_base_features: List[FeatureBase] = []
    for bf in feature.base_features:
        fb = feature_cache[bf.id]
        if bf.is_multioutput():
            idx = bf.idx
            # if its multioutput, you can index on the FeatureBase
            new_base_features.append(fb[idx])
        else:
            new_base_features.append(fb)

    return new_base_features


def to_transform_feature(
    feature: Feature,
    base_features: List[FeatureBase],
) -> FeatureBase:
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
    feature: Feature,
    dataframe: pd.DataFrame,
    cache: FeatureCache = {},
):
    def rfunc(feature: Feature) -> FeatureBase:
        # if feature has already been converted, return from cache
        if feature.id in cache:
            return cache[feature.id]

        # if depth is 0, we are at an origin featire
        if feature.depth == 0:
            fb = to_identity_feature(feature=feature, dataframe=dataframe)
            cache[feature.id] = fb
            return fb

        base_features = [rfunc(bf) for bf in feature.base_features]
        base_features = get_base_features(feature, cache)

        fb = to_transform_feature(feature, base_features)
        cache[feature.id] = fb
        return fb

    return rfunc(feature)


def convert_feature_list_to_featurebase_list(
    feature_list: List[Feature],
    dataframe: pd.DataFrame,
) -> List[FeatureBase]:
    feature_cache: FeatureCache = {}

    converted_features: List[FeatureBase] = []
    for feature in feature_list:
        if feature.is_multioutput():
            related_feature_ids = [f.id for f in feature.related_features]
            if any([x in feature_cache for x in related_feature_ids]):
                # feature base already created for related ids
                continue

        fb = convert_feature_to_featurebase(
            feature=feature,
            dataframe=dataframe,
            cache=feature_cache,
        )
        converted_features.append(fb)

    return converted_features
    # def rfunc(feature: Feature, depth=0) -> List[FeatureBase]:
    #     if feature.id in feature_cache:
    #         return feature_cache[feature.id]

    #     if feature.depth == 0:
    #         fb = IdentityFeature(dataframe.ww[feature.name])
    #         fb = cast(IdentityFeature, fb.rename(feature.name))
    #         feature_cache[feature.id] = [fb]
    #         return [fb]

    #     assert feature.primitive
    #     assert isinstance(
    #         feature.primitive,
    #         TransformPrimitive,
    #     ), "Only Transform Primitives"

    #     base_feature_sets = [rfunc(bf, depth=1) for bf in feature.base_features]

    #     out = []
    #     for base_features in product(*base_feature_sets):
    #         if feature.primitive.number_output_features > 1:
    #             assert (
    #                 len(feature.related_features)
    #                 == feature.primitive.number_output_features - 1
    #             )

    #             if any([f.id in feature_cache for f in feature.related_features]):
    #                 # if related id is already in cache, we already created this feature
    #                 continue

    #             # sort the features according to index to be in the right order
    #             sorted_features = sorted(
    #                 [f for f in feature.related_features] + [feature],
    #                 key=lambda x: x.idx,
    #             )
    #             names = [x.get_name() for x in sorted_features]
    #             fb = TransformFeature(base_features, feature.primitive)
    #             # raise
    #             fb.set_feature_names(names)
    #             feature_cache[feature.id] = [fb]

    #             if depth > 0:
    #                 out.extend(
    #                     [
    #                         fb[i]
    #                         for i in range(feature.primitive.number_output_features)
    #                     ],
    #                 )
    #             else:
    #                 out.append(fb)
    #         else:
    #             fb = TransformFeature(base_features, feature.primitive)

    #             # TODO(dreed): I think I need this if features are renamed
    #             # fb = fb.rename(feature.get_name())

    #             feature_cache[feature.id] = [fb]
    #             out.append(fb)

    #     return out

    # final_features = [rfunc(f) for f in feature_list]

    # return [item for sublist in final_features for item in sublist]
