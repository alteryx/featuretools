import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Type

from woodwork.logical_types import LogicalType

from featuretools.primitives.base.primitive_base import PrimitiveBase


@dataclass
class Feature:
    name: Optional[str]

    logical_type: Type[LogicalType]
    tags: Set[str] = field(default_factory=set)
    primitive: Optional[Type[PrimitiveBase]] = None
    base_columns: List[str] = field(default_factory=list)
    df_id: Optional[str] = None
    id: str = field(init=False)

    @staticmethod
    def hash(
        name: Optional[str],
        primitive: Optional[Type[PrimitiveBase]] = None,
        base_columns: List[str] = [],
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

            assert len(base_columns) > 0
            base_columns = base_columns
            if commutative:
                base_columns.sort()

            for c in base_columns:
                hash_msg.update(c.encode("utf-8"))

        else:
            assert name
            hash_msg.update(name.encode("utf-8"))

        return hash_msg.hexdigest()

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def _generate_hash(self) -> str:
        return self.hash(
            name=self.name,
            primitive=self.primitive,
            base_columns=self.base_columns,
            df_id=self.df_id,
        )

    def __post_init__(self):
        self.id = self._generate_hash()
