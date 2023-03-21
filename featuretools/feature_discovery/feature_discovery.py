from itertools import combinations, permutations, product
from typing import Dict, List, Type

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType
from woodwork.table_schema import TableSchema

from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives.base.primitive_base import PrimitiveBase


def index_input_set(input_set: List[ColumnSchema]):
    out = {}
    for c in input_set:
        lt = type(c.logical_type).__name__

        if lt == "NoneType":
            lt_key = None
        else:
            lt_key = lt

        tags = c.semantic_tags

        if len(tags) > 0:
            for tag in tags:
                if tag is not None:
                    key = f"{lt_key},{tag}" if lt_key is not None else tag
                    out[key] = out.get(key, 0) + 1

        elif lt_key is not None:
            out[lt_key] = out.get(lt_key, 0) + 1
        else:
            if "ANY" in out:
                out["ANY"] += 1
            else:
                out["ANY"] = 1

    return out


def get_features(
    col_groups: Dict[str, List[Feature]],
    input_set: List[ColumnSchema],
    commutative: bool,
):
    indexed_input_set = index_input_set(input_set)

    prod_iter = []
    for name, count in indexed_input_set.items():
        if name not in col_groups:
            return []

        features2 = col_groups[name]

        if commutative:
            prod_iter.append(combinations(features2, count))
        else:
            prod_iter.append(permutations(features2, count))

    out = product(*prod_iter)

    out3 = []
    for x in out:
        out2 = []
        for y in x:
            for z in y:
                out2.append(z)
        out3.append(out2)
    return out3


def group_features(features: List[Feature]) -> Dict[str, List[Feature]]:
    groups = {"ANY": []}
    for f in features:
        logical_type = f.logical_type
        lt_name = None
        if logical_type is not None:
            lt_name = logical_type.__name__
            groups.setdefault(lt_name, []).append(f)

        inferred_tags = (
            ww_type_system.str_to_logical_type(lt_name).standard_tags
            if lt_name
            else set()
        )
        for tag in inferred_tags.union(f.tags):
            groups.setdefault(tag, []).append(f)
            groups.setdefault(f"{lt_name},{tag}", []).append(f)

        groups["ANY"].append(f)

    return groups


def get_matching_columns(
    col_groups: Dict[str, List[Feature]],
    primitive: Type[PrimitiveBase],
):
    input_sets = primitive.input_types
    assert input_sets is not None
    if not isinstance(input_sets[0], list):
        input_sets = [primitive.input_types]

    commutative = primitive.commutative

    out3 = []
    for input_set in input_sets:
        assert input_set is not None
        out = get_features(
            col_groups=col_groups,
            input_set=input_set,
            commutative=commutative,
        )

        out3.extend(out)

    return out3


def get_primitive_return_type(primitive: Type[PrimitiveBase]) -> ColumnSchema:
    if primitive.return_type is None:
        return_type = primitive.input_types[0]
        if isinstance(return_type, list):
            return_type = return_type[0]
    else:
        return_type = primitive.return_type

    return return_type


def features_from_primitive(
    primitive: Type[PrimitiveBase],
    col_groups: Dict[str, List[Feature]],
):
    return_schema = get_primitive_return_type(primitive=primitive)
    assert isinstance(return_schema, ColumnSchema)

    output_logical_type = return_schema.logical_type

    output_tags = return_schema.semantic_tags
    assert isinstance(output_tags, set)

    features = []
    feature_sets = get_matching_columns(
        col_groups=col_groups,
        primitive=primitive,
    )
    for feature_set in feature_sets:
        if output_logical_type is None:
            # TODO: big hack here to get a firm return type. I'm not sure if this works
            output_logical_type = feature_set[0].logical_type

        # TODO: a hack to instantiate primitive to get access to generate_name
        prim_instance = primitive()
        features.append(
            Feature(
                name=prim_instance.generate_name([x.name for x in feature_set]),
                logical_type=output_logical_type,
                tags=output_tags,
                primitive=primitive,
                base_features=feature_set,
            ),
        )
    return features


def my_dfs(schema: TableSchema, primitives: List[Type[PrimitiveBase]]) -> List[Feature]:
    features = []
    for col_name, column_schema in schema.columns.items():
        assert isinstance(column_schema, ColumnSchema)

        logical_type = column_schema.logical_type
        assert logical_type
        assert issubclass(type(logical_type), LogicalType)

        tags = column_schema.semantic_tags
        assert isinstance(tags, set)

        # TODO: ignorning index columns. Think more about this and put this in a differnt location
        if "index" in tags:
            continue

        features.append(
            Feature(
                name=col_name,
                logical_type=type(logical_type),
                tags=tags,
            ),
        )

    # Group Columns by LogicalType, Tag, and combination
    col_groups = group_features(features=features)

    for primitive in primitives:
        features_ = features_from_primitive(primitive=primitive, col_groups=col_groups)
        features.extend(features_)

    return features
