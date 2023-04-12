from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType

from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.utils import (
    PrimitivesDeserializer,
    get_all_logical_types,
    get_all_primitives,
    serialize_primitive,
)

ANY = "ANY"

primitives_map = get_all_primitives()
logical_types_map = get_all_logical_types()

inferred_tag_map: Dict[Union[str, None], Set[str]] = {
    k: ww_type_system.str_to_logical_type(k).standard_tags
    for k in logical_types_map.keys()
}
inferred_tag_map[None] = set()


@total_ordering
@dataclass
class LiteFeature:
    name: Optional[str] = None
    logical_type: Optional[Type[LogicalType]] = None
    tags: Set[str] = field(default_factory=set)
    primitive: Optional[PrimitiveBase] = None
    base_features: List[LiteFeature] = field(default_factory=list)
    df_id: Optional[str] = None

    id: str = field(init=False)
    _gen_name: str = field(init=False)
    n_output_features: int = 1

    depth = 0
    related_features: Set[LiteFeature] = field(default_factory=set)
    idx: int = 0

    @staticmethod
    def hash(
        name: Optional[str],
        primitive: Optional[PrimitiveBase] = None,
        base_features: List[LiteFeature] = [],
        df_id: Optional[str] = None,
        idx: int = 0,
    ):
        hash_msg = hashlib.sha256()

        if df_id:
            hash_msg.update(df_id.encode("utf-8"))

        if primitive:
            # TODO: hashing should be on primitive
            hash_msg.update(hash_primitive(primitive)[0].encode("utf-8"))
            commutative = primitive.commutative
            assert (
                len(base_features) > 0
            ), "there must be base features if give a primitive"
            base_columns = base_features
            if commutative:
                base_features.sort()

            for c in base_columns:
                hash_msg.update(c.id.encode("utf-8"))

        else:
            assert name
            hash_msg.update(name.encode("utf-8"))

        hash_msg.update(str(idx).encode("utf-8"))

        return hash_msg.hexdigest()

    def __eq__(self, other: LiteFeature):
        return self.id == other.id

    def __lt__(self, other: LiteFeature):
        return self.id < other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def _generate_hash(self) -> str:
        return self.hash(
            name=self.name,
            primitive=self.primitive,
            base_features=self.base_features,
            df_id=self.df_id,
            idx=self.idx,
        )

    def get_primitive_name(self) -> Union[str, None]:
        return self.primitive.name if self.primitive else None

    def get_dependencies(self, deep=False) -> List[LiteFeature]:
        flattened_dependencies = []
        for f in self.base_features:
            flattened_dependencies.append(f)

            if deep:
                dependencies = f.get_dependencies()
                if isinstance(dependencies, list):
                    flattened_dependencies.extend(dependencies)
                else:
                    flattened_dependencies.append(dependencies)
        return flattened_dependencies

    def get_origin_features(self) -> List[LiteFeature]:
        all_dependencies = self.get_dependencies(deep=True)
        return [f for f in all_dependencies if f.depth == 0]

    def __post_init__(self):
        self.id = self._generate_hash()

        if self.primitive:
            assert isinstance(self.primitive, PrimitiveBase)
            self.n_output_features = self.primitive.number_output_features
            self.depth = max([x.depth for x in self.base_features]) + 1
            self._gen_name = self.primitive.generate_name(
                [x.get_name() for x in self.base_features],
            )

        elif self.name is None:
            raise Exception("Name must be given if origin feature")
        else:
            self._gen_name = self.name

        # TODO(dreed): find a better way to do this
        if self.logical_type is not None and "index" not in self.tags:
            logical_type_name = self.logical_type.__name__

            inferred_tags = inferred_tag_map[logical_type_name]
            self.tags = self.tags | inferred_tags

    @property
    def column_schema(self) -> ColumnSchema:
        return ColumnSchema(logical_type=self.logical_type, semantic_tags=self.tags)

    def rename(self, name: str):
        self.name = name

    def get_name(self) -> str:
        if self.name:
            return self.name
        elif len(self.related_features) > 0:
            return f"{self._gen_name}[{self.idx}]"
        return self._gen_name

    def get_depth(self) -> int:
        return self.depth

    def dependendent_primitives(self) -> Set[Type[PrimitiveBase]]:
        dependent_features = self.get_dependencies(deep=True)
        dependent_primitives = {
            type(f.primitive) for f in dependent_features if f.primitive
        }
        if self.primitive:
            dependent_primitives.add(type(self.primitive))
        return dependent_primitives

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "logical_type": self.logical_type.__name__ if self.logical_type else None,
            "tags": list(self.tags),
            "primitive": hash_primitive(self.primitive)[0] if self.primitive else None,
            "base_features": [x.id for x in self.base_features],
            "df_id": self.df_id,
            "id": self.id,
            "related_features": [x.id for x in self.related_features],
            "idx": self.idx,
        }

    def is_multioutput(self) -> bool:
        return len(self.related_features) > 0

    def copy(self) -> LiteFeature:
        copied_feature = LiteFeature(
            name=self.name,
            logical_type=self.logical_type,
            tags=self.tags,
            primitive=self.primitive,
            base_features=[x.copy() for x in self.base_features],
            df_id=self.df_id,
            idx=self.idx,
            related_features=self.related_features,
        )

        return copied_feature


def hash_primitive(primitive: PrimitiveBase) -> Tuple[str, Dict[str, Any]]:
    hash_msg = hashlib.sha256()
    primitive_name = primitive.name
    assert isinstance(primitive_name, str)
    primitive_dict = serialize_primitive(primitive)
    primitive_json = json.dumps(primitive_dict).encode("utf-8")
    hash_msg.update(primitive_json)
    key = hash_msg.hexdigest()
    return (key, primitive_dict)


class FeatureCollection:
    # TODO: this could be sped up by not doing all this work in the initializer and intead creating an index method
    def __init__(self, features: List[LiteFeature]):
        self.all_features: List[LiteFeature] = sorted(features)
        self.by_logical_type: Dict[
            Union[Type[LogicalType], None],
            Set[LiteFeature],
        ] = {}
        self.by_tag: Dict[str, Set[LiteFeature]] = {}
        self.by_origin_feature: Dict[LiteFeature, Set[LiteFeature]] = {}
        self.by_depth: Dict[int, Set[LiteFeature]] = {}
        self.by_name: Dict[str, LiteFeature] = {}

        for feature in features:
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

    def __eq__(self, other: FeatureCollection) -> bool:
        return all([x == y for x, y in zip(self.all_features, other.all_features)])
