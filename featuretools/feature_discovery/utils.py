import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, Tuple

from woodwork.column_schema import ColumnSchema

from featuretools.feature_discovery.type_defs import ANY
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.utils import (
    get_all_logical_type_names,
    get_all_primitives,
    serialize_primitive,
)

primitives_map = get_all_primitives()
logical_types_map = get_all_logical_type_names()


def column_schema_to_keys(column_schema: ColumnSchema) -> str:
    """
    Generate a hashing key from a Columns Schema. For example:
    - ColumnSchema(logical_type=Double) -> "Double"
    - ColumnSchema(semantic_tags={"index"}) -> "index"
    - ColumnSchema(logical_type=Double, semantic_tags={"index", "other"}) -> "Double,index,other"

    Args:
        column_schema (ColumnSchema):

    Returns:
        str: hashing key
    """
    logical_type = column_schema.logical_type
    tags = column_schema.semantic_tags
    lt_key = None
    if logical_type:
        lt_key = type(logical_type).__name__

    tags = sorted(tags)
    if len(tags) > 0:
        tag_key = ",".join(tags)
        return f"{lt_key},{tag_key}" if lt_key is not None else tag_key

    elif lt_key is not None:
        return lt_key
    else:
        return ANY


@lru_cache(maxsize=None)
def hash_primitive(primitive: PrimitiveBase) -> Tuple[str, Dict[str, Any]]:
    hash_msg = hashlib.sha256()
    primitive_name = primitive.name
    assert isinstance(primitive_name, str)
    primitive_dict = serialize_primitive(primitive)
    primitive_json = json.dumps(primitive_dict).encode("utf-8")
    hash_msg.update(primitive_json)
    key = hash_msg.hexdigest()
    return (key, primitive_dict)


def get_primitive_return_type(primitive: PrimitiveBase) -> ColumnSchema:
    """
    Get Return type from a primitive

    Args:
        primitive (PrimitiveBase)

    Returns:
        ColumnSchema
    """
    if primitive.return_type:
        return primitive.return_type
    return_type = primitive.input_types[0]
    if isinstance(return_type, list):
        return_type = return_type[0]
    return return_type
