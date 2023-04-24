from __future__ import annotations

import hashlib
from dataclasses import field
from functools import total_ordering
from typing import Any, Dict, List, Optional, Set, Type, Union

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType

from featuretools.feature_discovery.utils import (
    get_primitive_return_type,
    hash_primitive,
)
from featuretools.primitives.base.primitive_base import PrimitiveBase


@total_ordering
class LiteFeature:
    _name: Optional[str] = None
    _alias: Optional[str] = None

    _logical_type: Optional[Type[LogicalType]] = None
    _tags: Set[str] = field(default_factory=set)
    _primitive: Optional[PrimitiveBase] = None
    _base_features: List[LiteFeature] = field(default_factory=list)
    _df_id: Optional[str] = None

    _id: str
    _n_output_features: int = 1

    _depth = 0
    _related_features: Set[LiteFeature]
    _idx: int = 0

    def __init__(
        self,
        name: Optional[str] = None,
        logical_type: Optional[Type[LogicalType]] = None,
        tags: Optional[Set[str]] = None,
        primitive: Optional[PrimitiveBase] = None,
        base_features: Optional[List[LiteFeature]] = None,
        df_id: Optional[str] = None,
        related_features: Optional[Set[LiteFeature]] = None,
        idx: Optional[int] = None,
    ):
        self._logical_type = logical_type
        self._tags = tags if tags else set()
        self._primitive = primitive
        self._base_features = base_features if base_features else []
        self._df_id = df_id
        self._idx = idx if idx is not None else 0
        self._related_features = related_features if related_features else set()

        if self._primitive:
            if not isinstance(self._primitive, PrimitiveBase):
                raise ValueError("primitive input must be of type PrimitiveBase")

            if len(self.base_features) == 0:
                raise ValueError("there must be base features if given a primitive")

            if self._primitive.commutative:
                self._base_features = sorted(self._base_features)

            self._n_output_features = self._primitive.number_output_features
            self._depth = max([x.depth for x in self.base_features]) + 1

            if name:
                self._alias = name

            self._name = self._primitive.generate_name(
                [x.name for x in self.base_features],
            )

            return_column_schema = get_primitive_return_type(self._primitive)
            self._logical_type = (
                type(return_column_schema.logical_type)
                if return_column_schema.logical_type
                else None
            )

            self._tags = return_column_schema.semantic_tags

        else:
            if name is None:
                raise TypeError("Name must be given if origin feature")

            if self._logical_type is None:
                raise TypeError("Logical Type must be given if origin feature")

            self._name = name

        if self._logical_type is not None and "index" not in self._tags:
            self._tags = self._tags | self._logical_type.standard_tags

        self._id = self._generate_hash()

    @property
    def name(self):
        if self._alias:
            return self._alias
        elif self.is_multioutput():
            return f"{self._name}[{self.idx}]"
        return self._name

    @name.setter
    def name(self, _):
        raise AttributeError("name is immutable")

    def set_alias(self, value: Union[str, None]):
        self._alias = value

    @property
    def non_indexed_name(self):
        if not self.is_multioutput():
            raise ValueError("only used on multioutput features")
        return self._name

    @property
    def logical_type(self):
        return self._logical_type

    @logical_type.setter
    def logical_type(self, _):
        raise AttributeError("logical_type is immutable")

    @property
    def tags(self):
        return self._tags.copy()

    @tags.setter
    def tags(self, _):
        raise AttributeError("tags is immutable")

    @property
    def primitive(self):
        return self._primitive

    @primitive.setter
    def primitive(self, _):
        raise AttributeError("primitive is immutable")

    @property
    def base_features(self):
        return self._base_features

    @base_features.setter
    def base_features(self, _):
        raise AttributeError("base_features are immutable")

    @property
    def df_id(self):
        return self._df_id

    @df_id.setter
    def df_id(self, _):
        raise AttributeError("df_id is immutable")

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, _):
        raise AttributeError("id is immutable")

    @property
    def n_output_features(self):
        return self._n_output_features

    @n_output_features.setter
    def n_output_features(self, _):
        raise AttributeError("n_output_features is immutable")

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, _):
        raise AttributeError("depth is immutable")

    @property
    def related_features(self):
        return self._related_features.copy()

    @related_features.setter
    def related_features(self, value: Set[LiteFeature]):
        self._related_features = value

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, _):
        raise AttributeError("idx is immutable")

    @staticmethod
    def hash(
        name: Optional[str],
        primitive: Optional[PrimitiveBase] = None,
        base_features: List[LiteFeature] = [],
        df_id: Optional[str] = None,
        idx: int = 0,
    ):
        hash_msg = hashlib.sha256()

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
            if df_id:
                hash_msg.update(df_id.encode("utf-8"))

        hash_msg.update(str(idx).encode("utf-8"))

        return hash_msg.hexdigest()

    def __eq__(self, other: LiteFeature):
        return self._id == other._id

    def __lt__(self, other: LiteFeature):
        return self._id < other._id

    def __ne__(self, other):
        return self._id != other._id

    def __hash__(self):
        return hash(self._id)

    def _generate_hash(self) -> str:
        return self.hash(
            name=self._name,
            primitive=self._primitive,
            base_features=self._base_features,
            df_id=self._df_id,
            idx=self._idx,
        )

    def get_primitive_name(self) -> Union[str, None]:
        return self._primitive.name if self._primitive else None

    def get_dependencies(self, deep=False) -> List[LiteFeature]:
        flattened_dependencies = []
        for f in self._base_features:
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
        return [f for f in all_dependencies if f._depth == 0]

    @property
    def column_schema(self) -> ColumnSchema:
        return ColumnSchema(logical_type=self.logical_type, semantic_tags=self.tags)

    def dependent_primitives(self) -> Set[Type[PrimitiveBase]]:
        dependent_features = self.get_dependencies(deep=True)
        dependent_primitives = {
            type(f._primitive) for f in dependent_features if f._primitive
        }
        if self._primitive:
            dependent_primitives.add(type(self._primitive))
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
        return len(self._related_features) > 0

    def copy(self) -> LiteFeature:
        copied_feature = LiteFeature(
            name=self._name,
            logical_type=self._logical_type,
            tags=self._tags.copy(),
            primitive=self._primitive,
            base_features=[f.copy() for f in self._base_features],
            df_id=self._df_id,
            idx=self._idx,
            related_features=self._related_features.copy(),
        )

        copied_feature.set_alias(self._alias)

        return copied_feature

    def __repr__(self) -> str:
        name = f"name='{self.name}'"
        logical_type = f"logical_type={self.logical_type}"
        tags = f"tags={self.tags}"
        primitive = f"primitive={self.get_primitive_name()}"
        return f"LiteFeature({name}, {logical_type}, {tags}, {primitive})"
