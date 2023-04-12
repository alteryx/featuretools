import hashlib
import json
from typing import Any, Dict, List, Set, Tuple, Union

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema

from featuretools.feature_discovery.type_defs import ANY
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.utils import (
    get_all_logical_types,
    get_all_primitives,
    serialize_primitive,
)

primitives_map = get_all_primitives()
logical_types_map = get_all_logical_types()

inferred_tag_map: Dict[Union[str, None], Set[str]] = {
    k: ww_type_system.str_to_logical_type(k).standard_tags
    for k in logical_types_map.keys()
}
inferred_tag_map[None] = set()


def column_schema_to_keys(column_schema: ColumnSchema) -> List[str]:
    """
    Generate hashing keys from Columns Schema. For example:
    - ColumnSchema(logical_type=Double) -> ["Double"]
    - ColumnSchema(semantic_tags={"index"}) -> ["index"]
    - ColumnSchema(logical_type=Double, semantic_tags={"index", "other"}) -> ["Double,index", "Double,other"]

    Args:
        column_schema (ColumnSchema):

    Returns:
        List[str]
            List of hashing keys
    """
    keys: List[str] = []
    logical_type = column_schema.logical_type
    tags = column_schema.semantic_tags
    lt_key = None
    if logical_type:
        lt_key = type(logical_type).__name__

    tags = sorted(tags)
    if len(tags) > 0:
        tag_key = ",".join(tags)
        keys.append(f"{lt_key},{tag_key}" if lt_key is not None else tag_key)

    elif lt_key is not None:
        keys.append(lt_key)
    else:
        keys.append(ANY)

    return keys


def hash_primitive(primitive: PrimitiveBase) -> Tuple[str, Dict[str, Any]]:
    hash_msg = hashlib.sha256()
    primitive_name = primitive.name
    assert isinstance(primitive_name, str)
    primitive_dict = serialize_primitive(primitive)
    primitive_json = json.dumps(primitive_dict).encode("utf-8")
    hash_msg.update(primitive_json)
    key = hash_msg.hexdigest()
    return (key, primitive_dict)
