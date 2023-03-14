import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Type

from woodwork.logical_types import LogicalType

from featuretools.primitives.base.primitive_base import PrimitiveBase


@dataclass
class Feature:
    name: Optional[str]

    logical_type: Type[LogicalType]
    primitive: Optional[Type[PrimitiveBase]] = None
    tags: Set[str] = field(default_factory=set)
    base_columns: List[str] = field(default_factory=list)
    df_id: Optional[str] = None
    id: str = field(init=False)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def _generate_hash(self) -> str:
        hash_msg = hashlib.sha256()

        if self.df_id:
            hash_msg.update(self.df_id.encode("utf-8"))

        if self.primitive:
            primitive_name = self.primitive.name
            assert isinstance(primitive_name, str)
            commutative = self.primitive.commutative
            hash_msg.update(primitive_name.encode("utf-8"))

            assert len(self.base_columns) > 0
            base_columns = self.base_columns
            if commutative:
                base_columns.sort()

            for c in base_columns:
                hash_msg.update(c.encode("utf-8"))

        else:
            assert self.name
            hash_msg.update(self.name.encode("utf-8"))

        return hash_msg.hexdigest()

    def __post_init__(self):

        self.id = self._generate_hash()
