from itertools import combinations, permutations, product
from typing import Dict, List, Type, cast

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
    col_groups: Dict[str, List[str]],
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


def group_columns(columns: List[Feature]):
    groups = {"ANY": []}
    for c in columns:
        lt_name = c.logical_type.__name__
        groups.setdefault(lt_name, []).append(c.id)

        inferred_tags = ww_type_system.str_to_logical_type(lt_name).standard_tags
        for tag in inferred_tags.union(c.tags):
            groups.setdefault(tag, []).append(c.id)
            groups.setdefault(f"{lt_name},{tag}", []).append(c.id)

        groups["ANY"].append(c.id)

    return groups


def get_matching_columns(
    col_groups: Dict[str, List[str]],
    primitive: Type[PrimitiveBase],
):
    input_sets = primitive.input_types

    assert input_sets is not None
    if not isinstance(primitive.input_types[0], list):
        input_sets = [primitive.input_types]

    commutative = primitive.commutative

    out3 = []
    for input_set in input_sets:
        out = get_features(
            col_groups=col_groups,
            input_set=input_set,
            commutative=commutative,
        )

        out3.extend(out)

    return out3


def my_dfs(schema: TableSchema, primitives: List[Type[PrimitiveBase]]) -> List[Feature]:
    features = []
    for col_name, column_schema in schema.columns.items():
        assert isinstance(column_schema, ColumnSchema)

        logical_type = column_schema.logical_type
        assert logical_type
        assert issubclass(type(logical_type), LogicalType)

        tags = column_schema.semantic_tags
        assert isinstance(tags, set)

        features.append(
            Feature(
                name=col_name,
                logical_type=type(logical_type),
                tags=tags,
            ),
        )

    col_groups = group_columns(columns=features)

    for primitive in primitives:

        return_schema = cast(ColumnSchema, primitive.return_type)
        assert isinstance(return_schema, ColumnSchema)

        output_logical_type = return_schema.logical_type

        output_tags = return_schema.semantic_tags
        assert isinstance(output_tags, set)

        x = get_matching_columns(
            col_groups=col_groups,
            primitive=primitive,
        )
        for base_columns in x:
            base_features = []
            for bc in base_columns:
                base_features.append([x for x in features if x.id == bc][0])
            if output_logical_type is None:
                # TODO: big hack here to get a firm return type. I'm not sure if this works
                output_logical_type = base_features[0].logical_type

            # TODO: a hack to instantiate primitive to get access to generate_name
            p = primitive()
            features.append(
                Feature(
                    name=p.generate_name([x.name for x in base_features]),
                    logical_type=output_logical_type,
                    tags=output_tags,
                    primitive=primitive,
                    base_columns=base_columns,
                ),
            )

    return features
