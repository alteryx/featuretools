import json

from featuretools.version import __version__ as ft_version

SCHEMA_VERSION = "1.0.0"


def save_features(features, filepath):
    """Saves the features list as JSON to a specificed filepath.

    Args:
        features (list[:class:`.FeatureBase`]): List of Feature definitions.

        filepath (str): The location of where to save the features list. This
            must include the name of the file.

    Note:
        Features saved in one version of Featuretools are not guaranteed to work in another.
        After upgrading Featuretools, features may need to be generated again.

    Example:
        .. ipython:: python
            :suppress:

            from featuretools.tests.testing_utils import (
                make_ecommerce_entityset)
            import featuretools as ft
            es = make_ecommerce_entityset()
            import os

        .. code-block:: python

            f1 = ft.Feature(es["log"]["product_id"])
            f2 = ft.Feature(es["log"]["purchased"])
            f3 = ft.Feature(es["log"]["value"])

            features = [f1, f2, f3]

            filepath = os.path.join('/Home/features/', 'list')
            ft.save_features(features, filepath)
    .. seealso::
        :func:`.load_features`
    """
    FeaturesSerializer(features).save(filepath)


class FeaturesSerializer(object):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def to_dict(self):
        names_list = [feat.unique_name() for feat in self.feature_list]
        es = self.feature_list[0].entityset
        return {
            'schema_version': SCHEMA_VERSION,
            'ft_version': ft_version,
            'entityset': es.to_dictionary(),
            'feature_list': names_list,
            'feature_definitions': self._feature_definitions(),
        }

    def save(self, filepath):
        features_dict = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(features_dict, f)

    def _feature_definitions(self):
        all_features = {}

        for feature in self.feature_list:
            all_features[feature.unique_name()] = self._serialize_feature(feature)

            for dependency in feature.get_dependencies(deep=True):
                name = dependency.unique_name()
                if name not in all_features:
                    all_features[name] = self._serialize_feature(dependency)

        return all_features

    def _serialize_feature(self, feature):
        return {
            'type': type(feature).__name__,
            'dependencies': [dep.unique_name() for dep in feature.get_dependencies()],
            'arguments': feature.get_arguments(),
        }
