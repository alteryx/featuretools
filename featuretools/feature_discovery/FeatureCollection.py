from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

from woodwork.logical_types import LogicalType

from featuretools.feature_discovery.LiteFeature import LiteFeature
from featuretools.feature_discovery.utils import hash_primitive, logical_types_map
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.utils import (
    PrimitivesDeserializer,
)


class FeatureCollection:
    def __init__(self, features: List[LiteFeature]):
        self.all_features: List[LiteFeature] = features
        self.indexed = False
        self.sorted = False
        self._hash_key: Optional[str] = None

    def sort_features(self):
        if not self.sorted:
            self.all_features = sorted(self.all_features)
            self.sorted = True

    def __repr__(self):
        return f"<FeatureCollection ({self.hash_key[:5]}) n_features={len(self.all_features)} indexed={self.indexed}>"

    @property
    def hash_key(self) -> str:
        if self._hash_key is None:
            if not self.sorted:
                self.sort_features()
            self._set_hash()
        assert self._hash_key is not None
        return self._hash_key

    def _set_hash(self):
        hash_msg = hashlib.sha256()

        for feature in self.all_features:
            hash_msg.update(feature.id.encode("utf-8"))

        self._hash_key = hash_msg.hexdigest()
        return self

    def __hash__(self):
        return hash(self.hash_key)

    def __eq__(self, other: FeatureCollection) -> bool:
        return self.hash_key == other.hash_key

    def reindex(self) -> FeatureCollection:
        self.by_logical_type: Dict[
            Union[Type[LogicalType], None],
            Set[LiteFeature],
        ] = {}
        self.by_tag: Dict[str, Set[LiteFeature]] = {}
        self.by_origin_feature: Dict[LiteFeature, Set[LiteFeature]] = {}
        self.by_depth: Dict[int, Set[LiteFeature]] = {}
        self.by_name: Dict[str, LiteFeature] = {}
        self.by_key: Dict[str, List[LiteFeature]] = {}

        for feature in self.all_features:
            for key in feature.to_keys():
                self.by_key.setdefault(key, []).append(feature)

            logical_type = feature.logical_type
            self.by_logical_type.setdefault(logical_type, set()).add(feature)

            tags = feature.tags
            for tag in tags:
                self.by_tag.setdefault(tag, set()).add(feature)

            origin_features = feature.get_origin_features()
            for origin_feature in origin_features:
                self.by_origin_feature.setdefault(origin_feature, set()).add(feature)

            if feature.depth == 0:
                self.by_origin_feature.setdefault(feature, set()).add(feature)

            feature_name = feature.get_name()
            assert feature_name is not None
            assert feature_name not in self.by_name

            self.by_name[feature_name] = feature

        self.indexed = True

        return self

    def get_by_logical_type(self, logical_type: Type[LogicalType]) -> Set[LiteFeature]:
        return self.by_logical_type.get(logical_type, set())

    def get_by_tag(self, tag: str) -> Set[LiteFeature]:
        return self.by_tag.get(tag, set())

    def get_by_origin_feature(self, origin_feature: LiteFeature) -> Set[LiteFeature]:
        return self.by_origin_feature.get(origin_feature, set())

    def get_by_origin_feature_name(self, name: str) -> LiteFeature:
        feature = self.by_name.get(name)
        assert feature is not None
        return feature

    def get_dependencies_by_origin_name(self, name) -> Set[LiteFeature]:
        origin_feature = self.by_name[name]

        assert origin_feature, "no origin feature with that name exists"

        return self.by_origin_feature[origin_feature]

    def get_by_key(self, key: str) -> Set[LiteFeature]:
        return self.by_key.get(key, set())

    def flatten_features(self) -> Dict[str, LiteFeature]:
        all_features_dict: Dict[str, LiteFeature] = {}

        def rfunc(feature_list: List[LiteFeature]):
            for feature in feature_list:
                all_features_dict.setdefault(feature.id, feature)
                rfunc(feature.base_features)

        rfunc(self.all_features)
        return all_features_dict

    def flatten_primitives(self) -> Dict[str, Dict[str, Any]]:
        all_primitives_dict: Dict[str, Dict[str, Any]] = {}

        def rfunc(feature_list: List[LiteFeature]):
            for feature in feature_list:
                if feature.primitive:
                    key, prim_dict = hash_primitive(feature.primitive)
                    all_primitives_dict.setdefault(key, prim_dict)
                rfunc(feature.base_features)

        rfunc(self.all_features)
        return all_primitives_dict

    def to_dict(self):
        all_primitives_dict = self.flatten_primitives()
        all_features_dict = self.flatten_features()

        return {
            "primitives": all_primitives_dict,
            "feature_ids": [f.id for f in self.all_features],
            "all_features": {k: f.to_dict() for k, f in all_features_dict.items()},
        }

    @staticmethod
    def from_dict(input_dict):
        primitive_deserializer = PrimitivesDeserializer()

        primitives = {}
        for prim_key, prim_dict in input_dict["primitives"].items():
            primitive = primitive_deserializer.deserialize_primitive(
                prim_dict,
            )
            assert isinstance(primitive, PrimitiveBase)
            primitives[prim_key] = primitive

        hydrated_features: Dict[str, LiteFeature] = {}

        feature_ids: List[str] = cast(List[str], input_dict["feature_ids"])
        all_features: Dict[str, Any] = cast(Dict[str, Any], input_dict["all_features"])

        def hydrate_feature(feature_id: str) -> LiteFeature:
            if feature_id in hydrated_features:
                return hydrated_features[feature_id]

            feature_dict = all_features[feature_id]
            base_features = [hydrate_feature(x) for x in feature_dict["base_features"]]
            #

            logical_type = (
                logical_types_map[feature_dict["logical_type"]]
                if feature_dict["logical_type"]
                else None
            )

            hydrated_feature = LiteFeature(
                name=feature_dict["name"],
                logical_type=logical_type,
                tags=set(feature_dict["tags"]),
                primitive=primitives[feature_dict["primitive"]]
                if feature_dict["primitive"]
                else None,
                base_features=base_features,
                df_id=feature_dict["df_id"],
                related_features=set(),
                idx=feature_dict["idx"],
            )

            assert hydrated_feature.id == feature_dict["id"] == feature_id
            hydrated_features[feature_id] = hydrated_feature

            # need to link after features are stored on cache
            related_features = [
                hydrate_feature(x) for x in feature_dict["related_features"]
            ]
            hydrated_feature.related_features = set(related_features)

            return hydrated_feature

        return FeatureCollection([hydrate_feature(x) for x in feature_ids])