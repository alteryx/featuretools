from dask import dataframe as dd

from featuretools.feature_base import (
    AggregationFeature,
    FeatureOutputSlice,
    GroupByTransformFeature,
    TransformFeature
)
from featuretools.utils.gen_utils import Library, import_or_none, is_instance

ks = import_or_none('databricks.koalas')


def _categorize_features(features):
    """Categorize each feature by its primitive type in a set of primitives along with any dependencies"""
    transform = set()
    agg = set()
    groupby = set()
    where = set()
    explored = set()

    def get_feature_data(feature):
        if feature.get_name() in explored:
            return

        dependencies = []

        if isinstance(feature, FeatureOutputSlice):
            feature = feature.base_feature

        if isinstance(feature, AggregationFeature):
            if feature.where:
                where.add(feature.primitive.name)
            else:
                agg.add(feature.primitive.name)
        elif isinstance(feature, GroupByTransformFeature):
            groupby.add(feature.primitive.name)
        elif isinstance(feature, TransformFeature):
            transform.add(feature.primitive.name)

        feature_deps = feature.get_dependencies()
        if feature_deps:
            dependencies.extend(feature_deps)

        explored.add(feature.get_name())

        for dep in dependencies:
            get_feature_data(dep)

    for feature in features:
        get_feature_data(feature)

    return transform, agg, groupby, where


def get_unused_primitives(specified, used):
    """Get a list of unused primitives based on a list of specified primitives and a list of output features"""
    if not specified:
        return []
    specified = {primitive if isinstance(primitive, str) else primitive.name for primitive in specified}
    return sorted(list(specified.difference(used)))


def _get_entityset_type(entityset):
    if any(isinstance(entity.df, dd.DataFrame) for entity in entityset.entities):
        entityset_type = Library.DASK
    elif any(is_instance(entity.df, ks, 'DataFrame') for entity in entityset.entities):
        entityset_type = Library.KOALAS
    else:
        entityset_type = Library.PANDAS

    return entityset_type
