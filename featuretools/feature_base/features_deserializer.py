import json
import warnings

from featuretools.entityset.deserialize import (
    description_to_entityset as deserialize_es,
)
from featuretools.feature_base.feature_base import (
    AggregationFeature,
    DirectFeature,
    Feature,
    FeatureBase,
    FeatureOutputSlice,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from featuretools.primitives.utils import PrimitivesDeserializer
from featuretools.utils.s3_utils import get_transport_params, use_smartopen_features
from featuretools.utils.schema_utils import check_schema_version
from featuretools.utils.wrangle import _is_s3, _is_url


def load_features(features, profile_name=None, trusted_source=False):
    """Loads the features from a filepath, S3 path, URL, an open file, or a JSON formatted string.

    Args:
        features (str or :class:`.FileObject`): The file location of saved features.
        This must either be the name of the file, a JSON formatted string, or a readable file handle.

        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.

        trusted_source (bool): Whether the source of the features is trusted. If False and the
            entityset was saved using pickle format, a RuntimeWarning will be raised about the
            risk of arbitrary code execution. Defaults to False.

    Returns:
        features (list[:class:`.FeatureBase`]): Feature definitions list.

    Note:
        Features saved in one version of Featuretools or Python are not guaranteed to work in another.
        After upgrading Featuretools or Python, features may need to be generated again.

    Example:
        .. ipython:: python
            :suppress:

            import featuretools as ft
            import os

        .. code-block:: python

            # Option 1
            filepath = os.path.join('/Home/features/', 'list.json')
            features = ft.load_features(filepath)

            # Option 2
            filepath = os.path.join('/Home/features/', 'list.json')
            with open(filepath, 'r') as f:
                features = ft.load_features(f)

            # Option 3
            filepath = os.path.join('/Home/features/', 'list.json')
            with open(filepath, 'r') as :
                feature_str = f.read()
            features = ft.load_features(feature_str)


    .. seealso::
        :func:`.save_features`
    """
    return FeaturesDeserializer.load(features, profile_name, trusted_source).to_list()


class FeaturesDeserializer(object):
    FEATURE_CLASSES = {
        "AggregationFeature": AggregationFeature,
        "DirectFeature": DirectFeature,
        "Feature": Feature,
        "FeatureBase": FeatureBase,
        "GroupByTransformFeature": GroupByTransformFeature,
        "IdentityFeature": IdentityFeature,
        "TransformFeature": TransformFeature,
        "FeatureOutputSlice": FeatureOutputSlice,
    }

    def __init__(self, features_dict, trusted_source=False):
        self.features_dict = features_dict
        self._check_schema_version()
        if (
            features_dict.get("entityset", {}).get("format") == "pickle"
            and not trusted_source
        ):
            warnings.warn(
                "The entityset was saved using pickle format. Loading pickle files "
                "from untrusted sources can result in arbitrary code execution. "
                "Only set trusted_source=True if you trust the source of this file.",
                RuntimeWarning,
            )
        self.entityset = deserialize_es(features_dict["entityset"], trusted_source=trusted_source)
        self._deserialized_features = {}  # name -> feature
        primitive_deserializer = PrimitivesDeserializer()
        primitive_definitions = features_dict["primitive_definitions"]
        self._deserialized_primitives = {
            k: primitive_deserializer.deserialize_primitive(v)
            for k, v in primitive_definitions.items()
        }

    @classmethod
    def load(cls, features, profile_name, trusted_source=False):
        if isinstance(features, str):
            try:
                features_dict = json.loads(features)
            except ValueError:
                if _is_url(features) or _is_s3(features):
                    transport_params = None
                    if _is_s3(features):
                        transport_params = get_transport_params(profile_name)
                    features_dict = use_smartopen_features(
                        features,
                        transport_params=transport_params,
                    )
                else:
                    with open(features, "r") as f:
                        features_dict = json.load(f)
            return cls(features_dict, trusted_source)
        return cls(json.load(features), trusted_source)

    def to_list(self):
        feature_names = self.features_dict["feature_list"]
        return [self._deserialize_feature(name) for name in feature_names]

    def _deserialize_feature(self, feature_name):
        if feature_name in self._deserialized_features:
            return self._deserialized_features[feature_name]

        feature_dict = self.features_dict["feature_definitions"][feature_name]
        dependencies_list = feature_dict["dependencies"]
        primitive = None
        primitive_id = feature_dict["arguments"].get("primitive")
        if primitive_id is not None:
            primitive = self._deserialized_primitives[primitive_id]

        # Collect dependencies into a dictionary of name -> feature.
        dependencies = {
            dependency: self._deserialize_feature(dependency)
            for dependency in dependencies_list
        }

        type = feature_dict["type"]
        cls = self.FEATURE_CLASSES.get(type)
        if not cls:
            raise RuntimeError('Unrecognized feature type "%s"' % type)

        args = feature_dict["arguments"]
        feature = cls.from_dictionary(args, self.entityset, dependencies, primitive)

        self._deserialized_features[feature_name] = feature
        return feature

    def _check_schema_version(self):
        check_schema_version(self, "features")
