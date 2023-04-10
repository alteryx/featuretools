from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Dict, List, Optional, Set, Type, Union

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


@total_ordering
@dataclass
class Feature:
    name: Optional[str] = None
    logical_type: Optional[Type[LogicalType]] = None
    tags: Set[str] = field(default_factory=set)
    primitive: Optional[PrimitiveBase] = None
    base_features: List[Feature] = field(default_factory=list)
    df_id: Optional[str] = None

    id: str = field(init=False)
    _gen_name: str = field(init=False)
    n_output_features: int = 1

    depth = 0
    related_features: Set[Feature] = field(default_factory=set)
    idx: int = 0

    @staticmethod
    def hash(
        name: Optional[str],
        primitive: Optional[PrimitiveBase] = None,
        base_features: List[Feature] = [],
        df_id: Optional[str] = None,
        idx: int = 0,
    ):
        hash_msg = hashlib.sha256()

        if df_id:
            hash_msg.update(df_id.encode("utf-8"))

        if primitive:
            primitive_name = primitive.name
            assert isinstance(primitive_name, str)
            commutative = primitive.commutative
            hash_msg.update(json.dumps(serialize_primitive(primitive)).encode("utf-8"))

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

    def __eq__(self, other: Feature):
        return self.id == other.id

    def __lt__(self, other: Feature):
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

    def get_dependencies(self, deep=False) -> List[Feature]:
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

    def get_origin_features(self) -> List[Feature]:
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

            inferred_tags = (
                ww_type_system.str_to_logical_type(logical_type_name).standard_tags
                if logical_type_name
                else set()
            )

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
            "primitive": serialize_primitive(self.primitive)
            if self.primitive
            else None,
            "base_features": [x.to_dict() for x in self.base_features],
            "df_id": self.df_id,
            "id": self.id,
        }

    def is_multioutput(self) -> bool:
        return len(self.related_features) > 0

    def copy(self) -> Feature:
        copied_feature = Feature(
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

    @staticmethod
    def from_dict(input_dict: Dict) -> Feature:
        # TODO(dreed): can this be initialized at the module level?
        primitive_deserializer = PrimitivesDeserializer()
        base_features = [Feature.from_dict(x) for x in input_dict["base_features"]]

        if input_dict["primitive"]:
            primitive = primitive_deserializer.deserialize_primitive(
                input_dict["primitive"],
            )
            assert isinstance(primitive, PrimitiveBase)
        else:
            primitive = None

        logical_type = (
            logical_types_map[input_dict["logical_type"]]
            if input_dict["logical_type"]
            else None
        )

        hydrated_feature = Feature(
            name=input_dict["name"],
            logical_type=logical_type,
            tags=set(input_dict["tags"]),
            primitive=primitive,
            base_features=base_features,
            df_id=input_dict["df_id"],
        )

        assert hydrated_feature.id == input_dict["id"]

        return hydrated_feature


class FeatureCollection:
    def __init__(self, features: List[Feature]):
        self.all_features: List[Feature] = features
        self.by_logical_type: Dict[Union[Type[LogicalType], None], Set[Feature]] = {}
        self.by_tag: Dict[str, Set[Feature]] = {}
        self.by_origin_feature: Dict[Feature, Set[Feature]] = {}
        self.by_depth: Dict[int, Set[Feature]] = {}
        self.by_name: Dict[str, Feature] = {}

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

    def get_by_logical_type(self, logical_type: Type[LogicalType]) -> Set[Feature]:
        return self.by_logical_type.get(logical_type, set())

    def get_by_tag(self, tag: str) -> Set[Feature]:
        return self.by_tag.get(tag, set())

    def get_by_origin_feature(self, origin_feature: Feature) -> Set[Feature]:
        return self.by_origin_feature.get(origin_feature, set())

    def get_by_origin_feature_name(self, name: str) -> Feature:
        feature = self.by_name.get(name)
        assert feature is not None
        return feature

    def get_dependencies_by_origin_name(self, name) -> Set[Feature]:
        origin_feature = self.by_name[name]

        assert origin_feature, "no origin feature with that name exists"

        return self.by_origin_feature[origin_feature]

    def to_dict():
        pass

    def from_dict():
        pass
