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
from featuretools.feature_base.features_serializer import SCHEMA_VERSION
from featuretools.primitives.utils import PrimitivesDeserializer
from featuretools.utils.gen_utils import is_python_2

if is_python_2():
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest


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
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            features_dict = json.load(f)

        return cls(features_dict)

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
        current = SCHEMA_VERSION.split('.')
        saved = self.features_dict['schema_version'].split('.')
        error_text = ('Unable to load features. The schema version of the saved '
                      'features (%s) is greater than the latest supported (%s). '
                      'You may need to upgrade featuretools.'
                      % (self.features_dict['schema_version'], SCHEMA_VERSION))

        for c_num, s_num in zip_longest(current, saved, fillvalue=0):
            if c_num > s_num:
                break
            elif c_num < s_num:
                raise RuntimeError(error_text)
