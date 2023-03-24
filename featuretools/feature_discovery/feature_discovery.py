from itertools import combinations, permutations, product
from typing import Dict, List, Type

import woodwork.type_sys.type_system as ww_type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType
from woodwork.table_schema import TableSchema

from featuretools.feature_discovery.type_defs import ANY, Feature
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.tests.testing_utils.generate_fake_dataframe import flatten_list


def generate_hashing_keys_from_column_schema(column_schema: ColumnSchema) -> List[str]:
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

    if len(tags) > 0:
        for tag in tags:
            if tag is not None:
                keys.append(f"{lt_key},{tag}" if lt_key is not None else tag)

    elif lt_key is not None:
        keys.append(lt_key)
    else:
        keys.append(ANY)

    return keys


def index_column_set(column_set: List[ColumnSchema]) -> Dict[str, int]:
    """
    Indexes input set to find types of columns and the quantity of eatch

    Args:
        column_set (List(ColumnSchema)):
            List of Column types needed by associated primitive.

    Returns:
        Dict[str, int]
            A hashmap from key to int

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import get_features
            from woodwork.column_schema import ColumnSchema

            column_set = [ColumnSchema(semantic_tags={"numeric"}), ColumnSchema(semantic_tags={"numeric"})]
            indexed_column_set = index_column_set(column_set)
            {"numeric": 2}
    """
    out = {}
    for column_schema in column_set:
        for key in generate_hashing_keys_from_column_schema(column_schema):
            out[key] = out.get(key, 0) + 1
    return out


def get_features(
    feature_groups: Dict[str, List[Feature]],
    column_set: List[ColumnSchema],
    commutative: bool,
) -> List[List[Feature]]:
    """
    Calculates all Feature combinations using the given hashmap of existing features, and the input set of required columns.

    Args:
        feature_groups (Dict[str, List[Feature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        column_set (List(ColumnSchema)):
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

            feature_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }
            column_set = [ColumnSchema(semantic_tags={"numeric"}), ColumnSchema(semantic_tags={"numeric"})]
            features = get_features(col_groups, column_set, commutative=False)
    """
    indexed_column_set = index_column_set(column_set)

    prod_iter = []
    for name, count in indexed_column_set.items():
        if name not in feature_groups:
            return []

        features2 = feature_groups[name]

        if commutative:
            prod_iter.append(combinations(features2, count))
        else:
            prod_iter.append(permutations(features2, count))

    feature_combinations = product(*prod_iter)

    return [flatten_list(x) for x in feature_combinations]


def feature_to_keys(feature: Feature) -> List[str]:
    """
    Generate hashing keys from Feature. For example:
    - Feature("f1", Double) -> ['Double', 'numeric', 'Double,numeric', 'ANY']
    - Feature("f1", Datetime, {"time_index"}) -> ['Datetime', 'time_index', 'Datetime,time_index', 'ANY']
    - Feature("f1", Double, {"index", "other"}) -> ['Double', 'index', 'other', 'Double,index', 'Double,other', 'ANY']

    Args:
        feature (Feature):

    Returns:
        List[str]
            List of hashing keys
    """
    keys: List[str] = []
    logical_type = feature.logical_type
    logical_type_name = None
    if logical_type is not None:
        logical_type_name = logical_type.__name__
        keys.append(logical_type_name)

    inferred_tags = (
        ww_type_system.str_to_logical_type(logical_type_name).standard_tags
        if logical_type_name
        else set()
    )
    if "index" in feature.tags:
        all_tags = feature.tags
    else:
        all_tags = inferred_tags.union(feature.tags)

    for tag in all_tags:
        keys.append(tag)
        keys.append(f"{logical_type_name},{tag}")

    keys.append(ANY)
    return keys


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
    groups = {ANY: []}
    for feature in features:
        for key in feature_to_keys(feature=feature):
            groups.setdefault(key, []).append(feature)

    return groups


def get_matching_features(
    feature_groups: Dict[str, List[Feature]],
    primitive: Type[PrimitiveBase],
) -> List[List[Feature]]:
    """
    For a given primitive, find all feature sets that can be used to create new feature

    Args:
        feature_groups (Dict[str, List[Feature]]):
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

            feature_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }

            feature_sets = get_matching_features(col_groups, AddNumeric)

            [
                ["f1", "f2"],
                ["f1", "f3"],
                ["f2", "f3"]
            ]
    """
    column_sets = primitive.input_types
    assert column_sets is not None
    if not isinstance(column_sets[0], list):
        column_sets = [primitive.input_types]

    commutative = primitive.commutative

    feature_sets = []
    for column_set in column_sets:
        assert column_set is not None
        feature_sets_ = get_features(
            feature_groups=feature_groups,
            column_set=column_set,
            commutative=commutative,
        )

        feature_sets.extend(feature_sets_)

    return feature_sets


def get_primitive_return_type(primitive: Type[PrimitiveBase]) -> ColumnSchema:
    if primitive.return_type:
        return primitive.return_type
    return_type = primitive.input_types[0]
    if isinstance(return_type, list):
        return_type = return_type[0]
    return return_type


def features_from_primitive(
    feature_groups: Dict[str, List[Feature]],
    primitive: Type[PrimitiveBase],
) -> List[Feature]:
    """
    For a given primitive, creates all engineered features

    Args:
        feature_groups (Dict[str, List[Feature]]):
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

            feature_groups = {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            }

            feature_sets = features_from_primitive(feature_groups, AddNumeric)

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
    feature_sets = get_matching_features(
        feature_groups=feature_groups,
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
            feature_groups=feature_groups,
            primitive=primitive,
        )
        features.extend(features_)

    return features
