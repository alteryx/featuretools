from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Type, Union

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType

from featuretools.feature_discovery.type_defs import ANY
from featuretools.feature_discovery.utils import hash_primitive, inferred_tag_map
from featuretools.primitives.base.primitive_base import PrimitiveBase


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

    def to_keys(self) -> List[str]:
        """
        Generate hashing keys from LiteFeature. For example:
        - LiteFeature("f1", Double) -> ['Double', 'numeric', 'Double,numeric', 'ANY']
        - LiteFeature("f1", Datetime, {"time_index"}) -> ['Datetime', 'time_index', 'Datetime,time_index', 'ANY']
        - LiteFeature("f1", Double, {"index", "other"}) -> ['Double', 'index', 'other', 'Double,index', 'Double,other', 'ANY']

        Args:
            feature (LiteFeature):

        Returns:
            List[str]
                List of hashing keys
        """
        keys: List[str] = []
        logical_type = self.logical_type
        logical_type_name = None
        if logical_type is not None:
            logical_type_name = logical_type.__name__
            keys.append(logical_type_name)

        inferred_tags = inferred_tag_map[logical_type_name]

        if "index" in self.tags:
            all_tags = self.tags
        else:
            all_tags = inferred_tags.union(self.tags)

        all_tags = sorted(all_tags)

        tag_combinations = []

        # generate combinations of all lengths from 1 to the length of the input list
        for i in range(1, len(all_tags) + 1):
            # generate combinations of length i and append to the combinations_list
            for comb in combinations(all_tags, i):
                tag_combinations.append(list(comb))

        for tag_combination in tag_combinations:
            tags_key = ",".join(tag_combination)
            keys.append(tags_key)
            if logical_type_name:
                keys.append(f"{logical_type_name},{tags_key}")

        keys.append(ANY)
        return keys
