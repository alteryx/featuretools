from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Dict, List, Optional, Set, Type

from woodwork.logical_types import LogicalType

from featuretools.feature_base.feature_base import FeatureBase
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.utils import get_all_logical_types, get_all_primitives

ANY = "ANY"

primitives_map = get_all_primitives()
logical_types_map = get_all_logical_types()


@total_ordering
@dataclass
class Feature:
    name: Optional[str]

    logical_type: Optional[Type[LogicalType]] = None
    tags: Set[str] = field(default_factory=set)
    primitive: Optional[Type[PrimitiveBase]] = None
    base_features: List[Feature] = field(default_factory=list)
    df_id: Optional[str] = None
    id: str = field(init=False)

    @staticmethod
    def hash(
        name: Optional[str],
        primitive: Optional[Type[PrimitiveBase]] = None,
        base_features: List[Feature] = [],
        df_id: Optional[str] = None,
    ):
        hash_msg = hashlib.sha256()

        if df_id:
            hash_msg.update(df_id.encode("utf-8"))

        if primitive:
            primitive_name = primitive.name
            assert isinstance(primitive_name, str)
            commutative = primitive.commutative
            hash_msg.update(primitive_name.encode("utf-8"))

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
        )

    def __post_init__(self):
        self.id = self._generate_hash()

    def to_dict(self):
        return {
            "name": self.name,
            "logical_type": self.logical_type.__name__ if self.logical_type else None,
            "tags": list(self.tags),
            "primitive": self.primitive.__name__ if self.primitive else None,
            "base_features": [x.to_dict() for x in self.base_features],
            "df_id": self.df_id,
            "id": self.id,
        }

    @staticmethod
    def from_dict(input_dict: Dict) -> Feature:
        base_features = [Feature.from_dict(x) for x in input_dict["base_features"]]

        primitive = (
            primitives_map[input_dict["primitive"]] if input_dict["primitive"] else None
        )

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


def convert_featurebase_to_feature(feature: FeatureBase):
    base_features = [convert_featurebase_to_feature(x) for x in feature.base_features]

    name = feature.get_name()
    col_schema = feature.column_schema

    logical_type = col_schema.logical_type
    if logical_type is not None:
        assert issubclass(type(logical_type), LogicalType)
        logical_type = type(logical_type)

    tags = col_schema.semantic_tags

    primitive = type(feature.primitive)
    if primitive == PrimitiveBase:
        primitive = None

    return Feature(
        name=name,
        logical_type=logical_type,
        tags=tags,
        primitive=primitive,
        base_features=base_features,
        # TODO: replace this with dataframe name?
        df_id=None,
    )
