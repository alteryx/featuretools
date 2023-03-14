from itertools import combinations, permutations, product
from typing import Dict, List

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema

from featuretools.feature_discovery.type_defs import Feature


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


def get_matching_columns(col_groups: Dict[str, List[str]], primitive):
    input_sets = primitive.input_types
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


# def my_dfs(schema) -> List[Feature]:
#     columns = []
#     features = []
#     for row in schema:
#         columns.append(Column(
#             display_name=row["displayName"],
#             internal_name=row['internalName'],
#             logical_type=TEMPO_TO_WW_TYPE_MAP[row["logicalType"]],
#             tags=row['tags']
#         ))
#         features.append(Feature(
#             name=row["internalName"],
#             primitive_name="PrimitiveBase",
#             base_columns=[],
#             commutative=False
#         ))

#     col_groups = group_columns(columns=columns)
#     primitives = get_valid_tempo_transform_primitives("classification")

#     print('# primitives', len(primitives))

#     for p in primitives:
#         x = get_matching_columns(
#             col_groups=col_groups,
#             primitive=p
#         )
#         for y in x:
#             for z in y:
#                 features.append(Feature(
#                     name=None,
#                     primitive_name=p.__name__,
#                     base_columns=z,
#                     commutative=p.commutative
#                 ))

#     return features
