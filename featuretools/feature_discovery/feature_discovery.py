from itertools import combinations, permutations, product
from typing import Dict, List, Type

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType
from woodwork.table_schema import TableSchema

from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives.base.primitive_base import PrimitiveBase


def index_input_set(input_set: List[ColumnSchema]) -> Dict[str, int]:
    """
    Indexes input set to find types of columns and the quantity of eatch

    Args:
        input_set (List(ColumnSchema)):
            List of Column types needed by associated primitive.

    Returns:
        Dict[str, int]
            A hashmap from key to int

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_features
            from woodwork.column_schema import ColumnSchema

            input_set = [ColumnSchema(semantic_tags={"numeric"}), ColumnSchema(semantic_tags={"numeric"})]
            indexed_input_set = index_input_set(input_set)
            {"numeric": 2}
    """
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
                    # TODO: create a function that consistently manages this key type
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
) -> List[List[Feature]]:
    """
    Calculates all Feature combinations using the given hashmap of existing features, and the input set of required columns.

    Args:
        col_groups (Dict[str, List[Feature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        input_set (List(ColumnSchema)):
            List of Column types needed by associated primitive.
        commutative (bool):
            whether or not we need to use product or combinations to create feature sets.

    Returns:
        List[List[Feature]]
            A list of Feature sets.

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_features
            from woodwork.column_schema import ColumnSchema

            col_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }
            input_set = [ColumnSchema(semantic_tags={"numeric"}), ColumnSchema(semantic_tags={"numeric"})]
            features = get_features(col_groups, input_set, commutative=False)
    """
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

    feature_combinations = product(*prod_iter)

    return [[z for y in x for z in y] for x in feature_combinations]


def group_features(features: List[Feature]) -> Dict[str, List[Feature]]:
    """
    Groups all Features by logical_type, tags, and combination

    Args:
        features ( List[Feature]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").

    Returns:
        Dict[str, List[Feature]]
            Hashmap from key to list of features

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_features
            from woodwork.column_schema import ColumnSchema

            f1 = Feature('f1', Double)
            f2 = Feature('f2', Boolean)

            feature_groups = group_features([f1, f2])

            {
                "ANY": ["f1", "f2"],
                "Double": ["f1"],
                "numeric": ["f1"],
                "Double,numeric": ["f1"],
                "Boolean": ["f3"]
            }
    """
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
        if "index" in f.tags:
            all_tags = f.tags
        else:
            all_tags = inferred_tags.union(f.tags)

        for tag in all_tags:
            groups.setdefault(tag, []).append(f)
            groups.setdefault(f"{lt_name},{tag}", []).append(f)

        groups["ANY"].append(f)

    return groups


def get_matching_columns(
    col_groups: Dict[str, List[Feature]],
    primitive: Type[PrimitiveBase],
) -> List[List[Feature]]:
    """
    For a given primitive, find all feature sets that can be used to create new feature

    Args:
        col_groups (Dict[str, List[Feature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        primitive (Type[PrimitiveBase])

    Returns:
        List[List[Feature]]
            List of feature sets

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_matching_columns
            from woodwork.column_schema import ColumnSchema

            col_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }

            feature_sets = get_matching_columns(col_groups, AddNumeric)

            [
                ["f1", "f2"],
                ["f1", "f3"],
                ["f2", "f3"]
            ]
    """
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
    col_groups: Dict[str, List[Feature]],
    primitive: Type[PrimitiveBase],
) -> List[Feature]:
    """
    For a given primitive, creates all engineered features

    Args:
        col_groups (Dict[str, List[Feature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        primitive (Type[PrimitiveBase])

    Returns:
        List[List[Feature]]
            List of feature sets

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_matching_columns
            from woodwork.column_schema import ColumnSchema

            col_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }

            feature_sets = features_from_primitive(col_groups, AddNumeric)

            [
                ["f1", "f2"],
                ["f1", "f3"],
                ["f2", "f3"]
            ]
    """
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
            logical_type = feature_set[0].logical_type
        else:
            logical_type = type(output_logical_type)

        assert issubclass(logical_type, LogicalType)

        # TODO: a hack to instantiate primitive to get access to generate_name
        prim_instance = primitive()
        features.append(
            Feature(
                name=prim_instance.generate_name([x.name for x in feature_set]),
                logical_type=logical_type,
                tags=output_tags,
                primitive=primitive,
                base_features=feature_set,
            ),
        )
    return features


def my_dfs(schema: TableSchema, primitives: List[Type[PrimitiveBase]]) -> List[Feature]:
    """
    Calculates all Features for a given input woodwork table schema and list of primitives.

    Args:
        schema (TableSchema):
            Woodwork TableSchema object
        primitives (List[Type[PrimitiveBase]])
            List of primitive classes

    Returns:
        List[Feature]

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import my_dfs
            from featuretools.primitives import Absolute, IsNull
            import pandas as pd
            import woodwork as ww

            df = pd.DataFrame({
                "idx": [0,1,2,3],
                "f1": ["A", "B", "C", "D"],
                "f2": [1.2, 2.3, 3.4, 4.5]
            })

            df.ww.init()

            features = my_dfs(df.ww.schema, [Absolute, IsNull])

    """
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

    # Group Columns by LogicalType, Tag, and combination
    feature_groups = group_features(features=features)

    for primitive in primitives:
        features_ = features_from_primitive(
            col_groups=feature_groups,
            primitive=primitive,
        )
        features.extend(features_)

    return features
