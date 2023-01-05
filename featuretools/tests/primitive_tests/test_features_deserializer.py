import logging
from unittest.mock import patch

import pandas as pd
import pytest

from featuretools import (
    AggregationFeature,
    Feature,
    IdentityFeature,
    TransformFeature,
    __version__,
)
from featuretools.feature_base.features_deserializer import FeaturesDeserializer
from featuretools.primitives import (
    Count,
    Max,
    MultiplyNumericScalar,
    NMostCommon,
    NumberOfCommonWords,
    NumUnique,
)
from featuretools.primitives.utils import serialize_primitive
from featuretools.utils.schema_utils import FEATURES_SCHEMA_VERSION


def test_single_feature(es):
    feature = IdentityFeature(es["log"].ww["value"])
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [feature.unique_name()],
        "feature_definitions": {feature.unique_name(): feature.to_dictionary()},
        "primitive_definitions": {},
    }
    deserializer = FeaturesDeserializer(dictionary)

    expected = [feature]
    assert expected == deserializer.to_list()


def test_multioutput_feature(es):
    value = IdentityFeature(es["log"].ww["product_id"])
    threecommon = NMostCommon()
    num_unique = NumUnique()
    tc = Feature(value, parent_dataframe_name="sessions", primitive=threecommon)

    features = [tc, value]
    for i in range(3):
        features.append(
            Feature(
                tc[i],
                parent_dataframe_name="customers",
                primitive=num_unique,
            ),
        )
        features.append(tc[i])

    flist = [feat.unique_name() for feat in features]
    fd = [feat.to_dictionary() for feat in features]
    fdict = dict(zip(flist, fd))

    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": flist,
        "feature_definitions": fdict,
    }
    dictionary["primitive_definitions"] = {
        "0": serialize_primitive(threecommon),
        "1": serialize_primitive(num_unique),
    }

    dictionary["feature_definitions"][flist[0]]["arguments"]["primitive"] = "0"
    dictionary["feature_definitions"][flist[2]]["arguments"]["primitive"] = "1"
    dictionary["feature_definitions"][flist[4]]["arguments"]["primitive"] = "1"
    dictionary["feature_definitions"][flist[6]]["arguments"]["primitive"] = "1"
    deserializer = FeaturesDeserializer(dictionary).to_list()

    for i in range(len(features)):
        assert features[i].unique_name() == deserializer[i].unique_name()


def test_base_features_in_list(es):
    max_primitive = Max()
    value = IdentityFeature(es["log"].ww["value"])
    max_feat = AggregationFeature(value, "sessions", max_primitive)
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feat.unique_name(), value.unique_name()],
        "feature_definitions": {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    dictionary["primitive_definitions"] = {"0": serialize_primitive(max_primitive)}
    dictionary["feature_definitions"][max_feat.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    deserializer = FeaturesDeserializer(dictionary)

    expected = [max_feat, value]
    assert expected == deserializer.to_list()


def test_base_features_not_in_list(es):
    max_primitive = Max()
    mult_primitive = MultiplyNumericScalar(value=2)
    value = IdentityFeature(es["log"].ww["value"])
    value_x2 = TransformFeature(value, mult_primitive)
    max_feat = AggregationFeature(value_x2, "sessions", max_primitive)
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feat.unique_name()],
        "feature_definitions": {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value_x2.unique_name(): value_x2.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    dictionary["primitive_definitions"] = {
        "0": serialize_primitive(max_primitive),
        "1": serialize_primitive(mult_primitive),
    }
    dictionary["feature_definitions"][max_feat.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    dictionary["feature_definitions"][value_x2.unique_name()]["arguments"][
        "primitive"
    ] = "1"
    deserializer = FeaturesDeserializer(dictionary)

    expected = [max_feat]
    assert expected == deserializer.to_list()


@patch("featuretools.utils.schema_utils.FEATURES_SCHEMA_VERSION", "1.1.1")
@pytest.mark.parametrize(
    "hardcoded_schema_version, warns",
    [("2.1.1", True), ("1.2.1", True), ("1.1.2", True), ("1.0.2", False)],
)
def test_later_schema_version(es, caplog, hardcoded_schema_version, warns):
    def test_version(version, warns):
        if warns:
            warning_text = (
                "The schema version of the saved features"
                "(%s) is greater than the latest supported (%s). "
                "You may need to upgrade featuretools. Attempting to load features ..."
                % (version, "1.1.1")
            )
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, "warn")

    test_version(hardcoded_schema_version, warns)


@patch("featuretools.utils.schema_utils.FEATURES_SCHEMA_VERSION", "1.1.1")
@pytest.mark.parametrize(
    "hardcoded_schema_version, warns",
    [("0.1.1", True), ("1.0.1", False), ("1.1.0", False)],
)
def test_earlier_schema_version(es, caplog, hardcoded_schema_version, warns):
    def test_version(version, warns):
        if warns:
            warning_text = (
                "The schema version of the saved features"
                "(%s) is no longer supported by this version "
                "of featuretools. Attempting to load features ..." % version
            )
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, "log")

    test_version(hardcoded_schema_version, warns)


def test_unknown_feature_type(es):
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": ["feature_1"],
        "feature_definitions": {
            "feature_1": {"type": "FakeFeature", "dependencies": [], "arguments": {}},
        },
        "primitive_definitions": {},
    }

    deserializer = FeaturesDeserializer(dictionary)

    with pytest.raises(RuntimeError, match='Unrecognized feature type "FakeFeature"'):
        deserializer.to_list()


def test_unknown_primitive_type(es):
    value = IdentityFeature(es["log"].ww["value"])
    max_feat = AggregationFeature(value, "sessions", Max)
    primitive_dict = serialize_primitive(Max())
    primitive_dict["type"] = "FakePrimitive"
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feat.unique_name(), value.unique_name()],
        "feature_definitions": {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
        "primitive_definitions": {"0": primitive_dict},
    }

    with pytest.raises(RuntimeError) as excinfo:
        FeaturesDeserializer(dictionary)

    error_text = 'Primitive "FakePrimitive" in module "%s" not found' % Max.__module__
    assert error_text == str(excinfo.value)


def test_unknown_primitive_module(es):
    value = IdentityFeature(es["log"].ww["value"])
    max_feat = AggregationFeature(value, "sessions", Max)
    primitive_dict = serialize_primitive(Max())
    primitive_dict["module"] = "fake.module"
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feat.unique_name(), value.unique_name()],
        "feature_definitions": {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
        "primitive_definitions": {"0": primitive_dict},
    }

    with pytest.raises(RuntimeError) as excinfo:
        FeaturesDeserializer(dictionary)

    error_text = 'Primitive "Max" in module "fake.module" not found'
    assert error_text == str(excinfo.value)


def test_feature_use_previous_pd_timedelta(es):
    value = IdentityFeature(es["log"].ww["id"])
    td = pd.Timedelta(12, "W")
    count_primitive = Count()
    count_feature = AggregationFeature(
        value,
        "customers",
        count_primitive,
        use_previous=td,
    )
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    dictionary["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    dictionary["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    deserializer = FeaturesDeserializer(dictionary)

    expected = [count_feature, value]
    assert expected == deserializer.to_list()


def test_feature_use_previous_pd_dateoffset(es):
    value = IdentityFeature(es["log"].ww["id"])
    do = pd.DateOffset(months=3)
    count_primitive = Count()
    count_feature = AggregationFeature(
        value,
        "customers",
        count_primitive,
        use_previous=do,
    )
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    dictionary["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    dictionary["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    deserializer = FeaturesDeserializer(dictionary)

    expected = [count_feature, value]
    assert expected == deserializer.to_list()

    value = IdentityFeature(es["log"].ww["id"])
    do = pd.DateOffset(months=3, days=2, minutes=30)
    count_feature = AggregationFeature(
        value,
        "customers",
        count_primitive,
        use_previous=do,
    )
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    dictionary["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    dictionary["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    deserializer = FeaturesDeserializer(dictionary)

    expected = [count_feature, value]
    assert expected == deserializer.to_list()


def test_word_set_in_number_of_common_words_is_deserialized_back_into_a_set(es):
    id_feat = IdentityFeature(es["log"].ww["comments"])
    number_of_common_words = NumberOfCommonWords(word_set={"hello", "my"})
    transform_feat = TransformFeature(id_feat, number_of_common_words)
    dictionary = {
        "ft_version": __version__,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [id_feat.unique_name(), transform_feat.unique_name()],
        "feature_definitions": {
            id_feat.unique_name(): id_feat.to_dictionary(),
            transform_feat.unique_name(): transform_feat.to_dictionary(),
        },
        "primitive_definitions": {"0": serialize_primitive(number_of_common_words)},
    }
    dictionary["feature_definitions"][transform_feat.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    deserializer = FeaturesDeserializer(dictionary)
    assert isinstance(
        deserializer.features_dict["primitive_definitions"]["0"]["arguments"][
            "word_set"
        ],
        set,
    )


def _check_schema_version(version, es, warning_text, caplog, warning_type=None):
    dictionary = {
        "ft_version": __version__,
        "schema_version": version,
        "entityset": es.to_dictionary(),
        "feature_list": [],
        "feature_definitions": {},
        "primitive_definitions": {},
    }

    if warning_type == "warn" and warning_text:
        with pytest.warns(UserWarning) as record:
            FeaturesDeserializer(dictionary)
        assert record[0].message.args[0] == warning_text
    elif warning_type == "log":
        logger = logging.getLogger("featuretools")
        logger.propagate = True
        FeaturesDeserializer(dictionary)
        if warning_text:
            assert warning_text in caplog.text
        else:
            assert not len(caplog.text)
        logger.propagate = False
