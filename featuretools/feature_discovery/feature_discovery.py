import concurrent.futures
import functools
import inspect
from itertools import combinations, permutations, product
from typing import Callable, Dict, Iterable, List, Set, Tuple, Type, Union, cast

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LogicalType
from woodwork.table_schema import TableSchema

from featuretools.feature_discovery.FeatureCollection import FeatureCollection
from featuretools.feature_discovery.LiteFeature import LiteFeature
from featuretools.feature_discovery.utils import column_schema_to_keys
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.tests.testing_utils.generate_fake_dataframe import flatten_list


def index_column_set(column_set: Tuple[ColumnSchema]) -> Dict[str, int]:
    """
    Indexes input set to find types of columns and the quantity of each

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
        for key in column_schema_to_keys(column_schema):
            out[key] = out.get(key, 0) + 1
    return out


def get_features(
    feature_collection: FeatureCollection,
    column_set: Tuple[ColumnSchema],
    commutative: bool,
) -> List[List[LiteFeature]]:
    """
    Calculates all LiteFeature combinations using the given hashmap of existing features, and the input set of required columns.

    Args:
        feature_groups (Dict[str, List[LiteFeature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        column_set (Tuple[ColumnSchema]):
            List of Column types needed by associated primitive.
        commutative (bool):
            whether or not we need to use product or combinations to create feature sets.

    Returns:
        List[List[LiteFeature]]
            A list of LiteFeature sets.

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
    for key, count in indexed_column_set.items():
        features2 = list(feature_collection.get_by_key(key))

        if commutative:
            prod_iter.append(combinations(features2, count))
        else:
            prod_iter.append(permutations(features2, count))

    feature_combinations = product(*prod_iter)

    return [flatten_list(x) for x in feature_combinations]


# def group_features(features: List[LiteFeature]) -> Dict[str, List[LiteFeature]]:
#     """
#     Groups all Features by logical_type, tags, and combination

#     Args:
#         features ( List[LiteFeature]):
#             Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
#             or combination (eg. "Double,index").

#     Returns:
#         Dict[str, List[LiteFeature]]
#             Hashmap from key to list of features

#     Examples:
#         .. code-block:: python

#             from featuretools.feature_discovery.feature_discovery import get_features
#             from woodwork.column_schema import ColumnSchema

#             f1 = LiteFeature('f1', Double)
#             f2 = LiteFeature('f2', Boolean)

#             feature_groups = group_features([f1, f2])

#             {
#                 "ANY": ["f1", "f2"],
#                 "Double": ["f1"],
#                 "numeric": ["f1"],
#                 "Double,numeric": ["f1"],
#                 "Boolean": ["f3"]
#             }
#     """
#     groups = {ANY: []}
#     for feature in features:
#         for key in feature_to_keys(feature=feature):
#             groups.setdefault(key, []).append(feature)

#     return groups


def primitive_to_columnsets(primitive: PrimitiveBase) -> List[List[ColumnSchema]]:
    column_sets = primitive.input_types
    assert column_sets is not None
    if not isinstance(column_sets[0], list):
        column_sets = [primitive.input_types]

    column_sets = cast(List[List[ColumnSchema]], column_sets)

    # Some primitives are commutative, yet have explicit versions of commutative pairs (eg. MultiplyNumericBoolean), which would create multiple versions, so this resolved that.
    if primitive.commutative:
        existing = set()
        uniq_column_sets = []
        for column_set in column_sets:
            key = "_".join(sorted([x.__repr__() for x in column_set]))
            if key not in existing:
                uniq_column_sets.append(column_set)
                existing.add(key)

        column_sets = uniq_column_sets

    return column_sets


def get_matching_features(
    feature_collection: FeatureCollection,
    primitive: PrimitiveBase,
) -> List[List[LiteFeature]]:
    """
    For a given primitive, find all feature sets that can be used to create new feature

    Args:
        feature_groups (Dict[str, List[LiteFeature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").
        primitive (Type[PrimitiveBase])

    Returns:
        List[List[LiteFeature]]
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
    column_sets = primitive_to_columnsets(primitive=primitive)

    commutative = primitive.commutative

    feature_sets = []
    for column_set in column_sets:
        assert column_set is not None
        feature_sets_ = get_features(
            feature_collection=feature_collection,
            column_set=tuple(column_set),
            commutative=commutative,
        )

        feature_sets.extend(feature_sets_)

    return feature_sets


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


def features_from_primitive(
    primitive: PrimitiveBase,
    feature_collection: FeatureCollection,
) -> List[LiteFeature]:
    """
    For a given primitive, creates all engineered features

    Args:
        primitive (Type[PrimitiveBase])
        feature_groups (Dict[str, List[LiteFeature]]):
            Hashmap from Key to List of Features. Key is either: LogicalType name (eg. "Double"), Semantic tag (eg. "index"),
            or combination (eg. "Double,index").

    Returns:
        List[List[LiteFeature]]
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

            feature_sets = features_from_primitive(AddNumeric, feature_groups)

            [
                ["f1", "f2"],
                ["f1", "f3"],
                ["f2", "f3"]
            ]
    """
    assert isinstance(primitive, PrimitiveBase)
    return_schema = get_primitive_return_type(primitive=primitive)
    assert isinstance(return_schema, ColumnSchema)

    output_logical_type = return_schema.logical_type

    output_tags = return_schema.semantic_tags
    assert isinstance(output_tags, set)

    features: List[LiteFeature] = []
    feature_sets = get_matching_features(
        feature_collection=feature_collection,
        primitive=primitive,
    )
    for feature_set in feature_sets:
        if output_logical_type is None:
            # TODO: big hack here to get a firm return type. I'm not sure if this works
            logical_type = feature_set[0].logical_type
        else:
            logical_type = type(output_logical_type)

        assert issubclass(logical_type, LogicalType)

        if primitive.number_output_features > 1:
            related_features: Set[LiteFeature] = set()
            for n in range(primitive.number_output_features):
                feature = LiteFeature(
                    logical_type=logical_type,
                    tags=output_tags,
                    primitive=primitive,
                    base_features=feature_set,
                    idx=n,
                )

                related_features.add(feature)

            for f in related_features:
                f.related_features = related_features - {f}
                features.append(f)
        else:
            features.append(
                LiteFeature(
                    name=primitive.generate_name([x.get_name() for x in feature_set]),
                    logical_type=logical_type,
                    tags=output_tags,
                    primitive=primitive,
                    base_features=feature_set,
                ),
            )
    return features


def schema_to_features(schema: TableSchema) -> List[LiteFeature]:
    """
    Convert a Woodwork Schema object to a list of origin features

    Args:
        schema (TableSchema):
            Woodwork TableSchema object

    Returns:
        List[LiteFeature]

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import schema_to_features
            from featuretools.primitives import Absolute, IsNull
            import pandas as pd
            import woodwork as ww

            df = pd.DataFrame({
                "idx": [0,1,2,3],
                "f1": ["A", "B", "C", "D"],
                "f2": [1.2, 2.3, 3.4, 4.5]
            })

            df.ww.init()

            features = schema_to_features(df.ww.schema)

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
            LiteFeature(
                name=col_name,
                logical_type=type(logical_type),
                tags=tags,
            ),
        )

    return features


def _check_inputs(
    input_features: Iterable[LiteFeature],
    primitives: Union[List[Type[PrimitiveBase]], List[PrimitiveBase]],
) -> Tuple[Iterable[LiteFeature], List[PrimitiveBase]]:
    if not hasattr(input_features, "__iter__"):
        raise ValueError("input_features must be an iterable of LiteFeature objects")

    for feature in input_features:
        if not isinstance(feature, LiteFeature):
            raise ValueError(
                "input_features must be an iterable of LiteFeature objects",
            )

    assert isinstance(primitives, List)

    primitive_instances: List[PrimitiveBase] = []
    for primitive in primitives:
        if inspect.isclass(primitive) and issubclass(primitive, PrimitiveBase):
            primitive_instances.append(primitive())
        elif isinstance(primitive, PrimitiveBase):
            primitive_instances.append(primitive)
        else:
            raise ValueError(
                "primitives must be a list of Primitive classes or Primitive instances",
            )

    return (input_features, primitive_instances)


def lite_dfs(
    input_features: Iterable[LiteFeature],
    primitives: Union[List[Type[PrimitiveBase]], List[PrimitiveBase]],
    parallelize=True,
) -> FeatureCollection:
    """
    Calculates all Features for a given input of features and a list of primitives.

    Args:
        origin_features (List[LiteFeature]):
            List of origin features
        primitives (List[Type[PrimitiveBase]])
            List of primitive classes

    Returns:
        List[LiteFeature]

    Examples:
        .. code-block:: python

            from featuretools.feature_discovery.feature_discovery import lite_dfs
            from featuretools.primitives import Absolute, IsNull
            import pandas as pd
            import woodwork as ww

            df = pd.DataFrame({
                "idx": [0,1,2,3],
                "f1": ["A", "B", "C", "D"],
                "f2": [1.2, 2.3, 3.4, 4.5]
            })

            df.ww.init()
            origin_features = schema_to_features(df.ww.schema)
            features = lite_dfs(origin_features, [Absolute, IsNull])

    """

    (input_features, primitives) = _check_inputs(input_features, primitives)

    features = [x.copy() for x in input_features]

    feature_collection = FeatureCollection(features=features)
    feature_collection.reindex()

    # # Group Columns by LogicalType, Tag, and combination
    # feature_groups = group_features(features=features)

    if parallelize:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            new_features = list(
                executor.map(
                    functools.partial(
                        features_from_primitive,
                        feature_collection=feature_collection,
                    ),
                    primitives,
                ),
            )

        features.extend(flatten_list(new_features))
    else:
        for primitive in primitives:
            features_ = features_from_primitive(
                primitive=primitive,
                feature_collection=feature_collection,
            )
            features.extend(features_)

    return FeatureCollection(features=features)


PredicateFuncType = Callable[[LiteFeature], bool]


def has_tag_predicate(tag: str) -> PredicateFuncType:
    return lambda feature: tag in feature.tags


def is_logical_type(logical_type: Type[LogicalType]) -> PredicateFuncType:
    return lambda feature: feature.logical_type == logical_type


def any_predicate_func(predicate_funcs: List[PredicateFuncType]) -> PredicateFuncType:
    return lambda feature: any([f(feature) for f in predicate_funcs])


def has_dependency(dependent_feature: LiteFeature) -> PredicateFuncType:
    return lambda feature: dependent_feature in feature.get_dependencies(deep=True)


def is_feature(other_feature: LiteFeature) -> PredicateFuncType:
    return lambda feature: feature == other_feature


def not_predicate(predicate_func: PredicateFuncType) -> PredicateFuncType:
    return lambda feature: not predicate_func(feature)


def filter_features(
    features: List[LiteFeature],
    predicate_funcs: List[PredicateFuncType],
):
    return [
        feature
        for feature in features
        if all(func(feature) for func in predicate_funcs)
    ]
