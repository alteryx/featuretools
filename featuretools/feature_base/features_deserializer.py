import json

from featuretools.entityset.deserialize import \
    description_to_entityset as deserialize_es
from featuretools.feature_base.feature_base import (
    AggregationFeature,
    DirectFeature,
    Feature,
    FeatureBase,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from featuretools.primitives.utils import PrimitivesDeserializer
from featuretools.utils.gen_utils import check_schema_version


def load_features(features):
    """Loads the features from a filepath, an open file, or a JSON formatted string.

    Args:
        features (str or :class:`.FileObject`): The location of where features has
        been saved which this must include the name of the file, or a JSON formatted
        string, or a readable file handle where the features have been saved.

    Returns:
        features (list[:class:`.FeatureBase`]): Feature definitions list.

    Note:
        Features saved in one version of Featuretools are not guaranteed to work in another.
        After upgrading Featuretools, features may need to be generated again.

    Example:
        .. ipython:: python
            :suppress:

            import featuretools as ft
            import os

        .. code-block:: python

            filepath = os.path.join('/Home/features/', 'list.json')
            ft.load_features(filepath)

            f = open(filepath, 'r')
            ft.load_features(f)

            feature_str = f.read()
            ft.load_features(feature_str)

    .. seealso::
        :func:`.save_features`
    """
    return FeaturesDeserializer.load(features).to_list()


class FeaturesDeserializer(object):
    FEATURE_CLASSES = {
        'AggregationFeature': AggregationFeature,
        'DirectFeature': DirectFeature,
        'Feature': Feature,
        'FeatureBase': FeatureBase,
        'GroupByTransformFeature': GroupByTransformFeature,
        'IdentityFeature': IdentityFeature,
        'TransformFeature': TransformFeature,
    }

    def __init__(self, features_dict):
        self.features_dict = features_dict
        self._check_schema_version()
        self.entityset = deserialize_es(features_dict['entityset'])
        self._deserialized_features = {}  # name -> feature
        self._primitives_deserializer = PrimitivesDeserializer()

    @classmethod
    def load(cls, features):
        if isinstance(features, str):
            try:
                features_dict = json.loads(features)
            except ValueError:
                with open(features, 'r') as f:
                    features_dict = json.load(f)
            return cls(features_dict)
        return cls(json.load(features))

    def to_list(self):
        feature_names = self.features_dict['feature_list']
        return [self._deserialize_feature(name) for name in feature_names]

    def _deserialize_feature(self, feature_name):
        if feature_name in self._deserialized_features:
            return self._deserialized_features[feature_name]

        feature_dict = self.features_dict['feature_definitions'][feature_name]
        dependencies_list = feature_dict['dependencies']

        # Collect dependencies into a dictionary of name -> feature.
        dependencies = {dependency: self._deserialize_feature(dependency)
                        for dependency in dependencies_list}

        type = feature_dict['type']
        cls = self.FEATURE_CLASSES.get(type)
        if not cls:
            raise RuntimeError('Unrecognized feature type "%s"' % type)

        args = feature_dict['arguments']
        feature = cls.from_dictionary(args, self.entityset, dependencies,
                                      self._primitives_deserializer)

        self._deserialized_features[feature_name] = feature
        return feature

    def _check_schema_version(self):
        check_schema_version(self, 'features')
