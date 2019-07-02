import json

from featuretools.version import __version__ as ft_version

SCHEMA_VERSION = "3.0.0"


def save_features(features, location=None):
    """Saves the features list as JSON to a specified filepath, writes to an open file, or
    returns the serialized features as a JSON string. If no file provided, returns a string.

    Args:
        features (list[:class:`.FeatureBase`]): List of Feature definitions.

        location (str or :class:`.FileObject`, optional): The location of where to save
            the features list which must include the name of the file,
            or a writeable file handle to write to. If location is None, will return a JSON string
            of the serialized features.
            Default: None

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

            filepath = os.path.join('/Home/features/', 'list.json')
            ft.save_features(features, filepath)

            f = open(filepath, 'w')
            ft.save_features(features, f)

            features_str = ft.save_features(features)
    .. seealso::
        :func:`.load_features`
    """
    return FeaturesSerializer(features).save(location)


class FeaturesSerializer(object):
    def __init__(self, feature_list):
        self.feature_list = feature_list
        self._features_dict = None

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

    def save(self, location):
        features_dict = self.to_dict()
        if location is None:
            return json.dumps(features_dict)
        if isinstance(location, str):
            with open(location, "w") as f:
                json.dump(features_dict, f)
        else:
            json.dump(features_dict, location)

    def _feature_definitions(self):
        if not self._features_dict:
            self._features_dict = {}

            for feature in self.feature_list:
                self._serialize_feature(feature)

        return self._features_dict

    def _serialize_feature(self, feature):
        name = feature.unique_name()

        if name not in self._features_dict:
            self._features_dict[feature.unique_name()] = feature.to_dictionary()

            for dependency in feature.get_dependencies(deep=True):
                name = dependency.unique_name()
                if name not in self._features_dict:
                    self._features_dict[name] = dependency.to_dictionary()
