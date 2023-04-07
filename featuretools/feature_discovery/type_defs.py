from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import product
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

import pandas as pd
import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import (
    FeatureBase,
    IdentityFeature,
    TransformFeature,
)
from featuretools.primitives import TransformPrimitive
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
    name: str = ""
    logical_type: Optional[Type[LogicalType]] = None
    tags: Set[str] = field(default_factory=set)
    primitive: Optional[PrimitiveBase] = None
    base_features: List[Feature] = field(default_factory=list)
    df_id: Optional[str] = None

    id: str = field(init=False)
    n_output_features: int = 1

    # _names: List[str] = field(init=False, default_factory=list)
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

            if self.name == "":
                # if name not provided, generate from primitive
                self.rename(
                    self.primitive.generate_name([x.name for x in self.base_features]),
                )
            else:
                self.rename(self.name)
        elif self.name == "":
            raise Exception("Name must be given if origin feature")
        else:
            self.rename(self.name)

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
        # if self.n_output_features > 1:
        #     self._names = [f"{name}[{idx}]" for idx in range(self.n_output_features)]
        # else:
        #     self._names = [name]

    def get_name(self) -> str:
        if len(self.related_features) > 0:
            return f"{self.name}[{self.idx}]"
        return self.name

    def get_depth(self) -> int:
        return self.depth

    # def get_feature_names(self) -> List[str]:
    #     return self._names

    # def split_feature(self) -> List[Feature]:
    #     if self.n_output_features == 1:
    #         return [self]

    #     out: List[Feature] = []
    #     for name in self._names:
    #         f = self.copy()
    #         f.n_output_features = 1
    #         f.rename(name)
    #         out.append(f)
    #     return out

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


def convert_featurebase_to_feature(feature: FeatureBase) -> Feature:
    base_features = [convert_featurebase_to_feature(x) for x in feature.base_features]

    name = feature.get_name()
    col_schema = feature.column_schema

    logical_type = col_schema.logical_type
    if logical_type is not None:
        assert issubclass(type(logical_type), LogicalType)
        logical_type = type(logical_type)

    tags = col_schema.semantic_tags

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


def convert_feature_to_featurebase(feature: Feature, es: EntitySet) -> FeatureBase:
    if feature.primitive is None:
        column_name = feature.name
        if feature.df_id is None:
            dataframe_names = list(es.dataframe_dict.keys())
            assert len(dataframe_names) == 1
            dataframe_name = dataframe_names[0]
        else:
            dataframe_name = feature.df_id
        return IdentityFeature(es[dataframe_name].ww[column_name])

    base_features = [
        convert_feature_to_featurebase(x, es) for x in feature.base_features
    ]

    return TransformFeature(base_features, feature.primitive)


def convert_feature_list_to_featurebase_list(
    dataframe: pd.DataFrame,
    feature_list: List[Feature],
) -> List[FeatureBase]:
    feature_cache: Dict[str, List[FeatureBase]] = {}

    def rfunc(feature: Feature, depth=0) -> List[FeatureBase]:
        if feature.id in feature_cache:
            return feature_cache[feature.id]

        if feature.depth == 0:
            fb = IdentityFeature(dataframe.ww[feature.name])
            fb = cast(IdentityFeature, fb.rename(feature.name))
            feature_cache[feature.id] = [fb]
            return [fb]

        assert feature.primitive
        assert isinstance(
            feature.primitive,
            TransformPrimitive,
        ), "Only Transform Primitives"

        base_feature_sets = [rfunc(bf, depth=1) for bf in feature.base_features]

        out = []
        for base_features in product(*base_feature_sets):
            if feature.primitive.number_output_features > 1:
                assert (
                    len(feature.related_features)
                    == feature.primitive.number_output_features - 1
                )

                if any([f.id in feature_cache for f in feature.related_features]):
                    # if related id is already in cache, we already created this feature
                    continue

                # sort the features according to index to be in the right order
                sorted_features = sorted(
                    [f for f in feature.related_features] + [feature],
                    key=lambda x: x.idx,
                )
                names = [x.get_name() for x in sorted_features]
                fb = TransformFeature(base_features, feature.primitive)
                # raise
                fb.set_feature_names(names)
                feature_cache[feature.id] = [fb]

                if depth > 0:
                    out.extend(
                        [
                            fb[i]
                            for i in range(feature.primitive.number_output_features)
                        ],
                    )
                else:
                    out.append(fb)
            else:
                fb = TransformFeature(base_features, feature.primitive)

                # TODO(dreed): I think I need this if features are renamed
                # fb = fb.rename(feature.get_name())

                feature_cache[feature.id] = [fb]
                out.append(fb)

        return out

    final_features = [rfunc(f) for f in feature_list]

    return [item for sublist in final_features for item in sublist]


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
