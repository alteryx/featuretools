import json

from .feature_base import (
    AggregationFeature,
    DirectFeature,
    Feature,
    FeatureBase,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from .features_serializer import SCHEMA_VERSION

from featuretools.entityset.deserialize import \
    description_to_entityset as deserialize_es


def load_features(filepath):
    """Loads the features from a filepath.

    Args:
        filepath (str): The location of where features has been saved.
            This must include the name of the file.

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

            filepath = os.path.join('/Home/features/', 'list')
            ft.load_features(filepath)
    .. seealso::
        :func:`.save_features`
    """
    return FeaturesDeserializer.load(filepath).to_list()


class FeaturesDeserializer(object):
    def __init__(self, features_dict):
        self.features_dict = features_dict
        self._check_version()
        self.entityset = deserialize_es(features_dict['entityset'])

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            features_dict = json.load(f)

        return cls(features_dict)

    def to_list(self):
        feature_names = self.features_dict['feature_list']
        deserialized = {}  # Dictionary of all features deserialized so far.
        return [self._deserialize_feature(name, deserialized) for name in feature_names]

    def _deserialize_feature(self, feature_name, deserialized):
        if feature_name in deserialized:
            return deserialized[feature_name]

        feature_dict = self.features_dict['feature_definitions'][feature_name]
        dependencies_list = feature_dict['dependencies']

        # Collect dependencies into a dictionary of name -> feature.
        dependencies = {dependency: self._deserialize_feature(dependency, deserialized)
                        for dependency in dependencies_list}

        cls = self._feature_class(feature_dict['type'])
        args = feature_dict['arguments']
        feature = cls.from_dictionary(args, self.entityset, dependencies)

        deserialized[feature_name] = feature
        return feature

    def _feature_class(self, class_name):
        if class_name == 'AggregationFeature':
            return AggregationFeature
        elif class_name == 'DirectFeature':
            return DirectFeature
        elif class_name == 'Feature':
            return Feature
        elif class_name == 'FeatureBase':
            return FeatureBase
        elif class_name == 'GroupByTransformFeature':
            return GroupByTransformFeature
        elif class_name == 'IdentityFeature':
            return IdentityFeature
        elif class_name == 'TransformFeature':
            return TransformFeature
        else:
            raise Exception("Unrecognized feature type")

    def _check_version(self):
        current = SCHEMA_VERSION.split('.')
        saved = self.features_dict['schema_version'].split('.')
        error_text = ('Unable to load features. The schema version of the saved '
                      'features (%s) is greater than the latest supported (%s). '
                      'You may need to upgrade featuretools.'
                      % (self.features_dict['schema_version'], SCHEMA_VERSION))

        if current[0] < saved[0] or (
           current[0] == saved[0] and ((current[1] < saved[1]) or (
                                       current[1] == saved[1] and current[2] < saved[2]))):
            raise RuntimeError(error_text)
