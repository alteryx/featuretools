import os

import boto3
import pandas as pd
import pytest
from pympler.asizeof import asizeof
from smart_open import open
from woodwork.column_schema import ColumnSchema

from featuretools import (
    AggregationFeature,
    DirectFeature,
    EntitySet,
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    dfs,
    feature_base,
    load_features,
    primitives,
    save_features,
)
from featuretools.feature_base import FeatureOutputSlice
from featuretools.feature_base.cache import feature_cache
from featuretools.feature_base.features_deserializer import FeaturesDeserializer
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import (
    Count,
    CumSum,
    Day,
    DistanceToHoliday,
    Haversine,
    IsIn,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    MultiplyNumericScalar,
    Negate,
    NMostCommon,
    NumberOfCommonWords,
    NumCharacters,
    NumUnique,
    NumWords,
    PercentTrue,
    Skew,
    Std,
    Sum,
    TransformPrimitive,
    Weekday,
    Year,
)
from featuretools.primitives.base import AggregationPrimitive
from featuretools.tests.testing_utils import check_names
from featuretools.utils.gen_utils import Library
from featuretools.version import ENTITYSET_SCHEMA_VERSION, FEATURES_SCHEMA_VERSION

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_feature_serialization_feature_schema_{}_entityset_schema_{}_2022_12_28.json".format(
    FEATURES_SCHEMA_VERSION,
    ENTITYSET_SCHEMA_VERSION,
)
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
TEST_CONFIG = "CheckConfigPassesOn"
TEST_KEY = "test_access_key_features"


@pytest.fixture(autouse=True)
def reset_dfs_cache():
    feature_cache.enabled = False
    feature_cache.clear_all()


def assert_features(original, deserialized):
    for feat_1, feat_2 in zip(original, deserialized):
        assert feat_1.unique_name() == feat_2.unique_name()
        assert feat_1.entityset == feat_2.entityset

        # IdentityFeature and DirectFeature objects do not have primitives, so
        # series library does not need to be compared
        if not (isinstance(feat_1, (IdentityFeature, DirectFeature))):
            assert feat_1.primitive.series_library == feat_2.primitive.series_library


def pickle_features_test_helper(es_size, features_original, dir_path):
    filepath = os.path.join(dir_path, "test_feature")

    save_features(features_original, filepath)
    features_deserializedA = load_features(filepath)
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    with open(filepath, "w") as f:
        save_features(features_original, f)
    features_deserializedB = load_features(open(filepath))
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    features = save_features(features_original)
    features_deserializedC = load_features(features)
    assert asizeof(features) < es_size

    features_deserialized_options = [
        features_deserializedA,
        features_deserializedB,
        features_deserializedC,
    ]
    for features_deserialized in features_deserialized_options:
        assert_features(features_original, features_deserialized)


def test_pickle_features(es, tmp_path):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )
    pickle_features_test_helper(asizeof(es), features_original, str(tmp_path))


def test_pickle_features_with_custom_primitive(pd_es, tmp_path):
    class NewMax(AggregationPrimitive):
        name = "new_max"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=pd_es,
        agg_primitives=["Last", "Mean", NewMax],
        features_only=True,
    )

    assert any([isinstance(feat.primitive, NewMax) for feat in features_original])
    pickle_features_test_helper(asizeof(pd_es), features_original, str(tmp_path))


def test_serialized_renamed_features(es):
    def serialize_name_unchanged(original):
        new_name = "MyFeature"
        original_names = original.get_feature_names()
        renamed = original.rename(new_name)
        new_names = (
            [new_name]
            if len(original_names) == 1
            else [new_name + "[{}]".format(i) for i in range(len(original_names))]
        )
        check_names(renamed, new_name, new_names)

        serializer = FeaturesSerializer([renamed])
        serialized = serializer.to_dict()

        deserializer = FeaturesDeserializer(serialized)
        deserialized = deserializer.to_list()[0]
        check_names(deserialized, new_name, new_names)

    identity_original = IdentityFeature(es["log"].ww["value"])
    assert identity_original.get_name() == "value"

    value = IdentityFeature(es["log"].ww["value"])

    primitive = primitives.Max()
    agg_original = AggregationFeature(value, "customers", primitive)
    assert agg_original.get_name() == "MAX(log.value)"

    direct_original = DirectFeature(
        IdentityFeature(es["customers"].ww["age"]),
        "sessions",
    )
    assert direct_original.get_name() == "customers.age"

    primitive = primitives.MultiplyNumericScalar(value=2)
    transform_original = TransformFeature(value, primitive)
    assert transform_original.get_name() == "value * 2"

    zipcode = IdentityFeature(es["log"].ww["zipcode"])
    primitive = CumSum()
    groupby_original = feature_base.GroupByTransformFeature(value, primitive, zipcode)
    assert groupby_original.get_name() == "CUM_SUM(value) by zipcode"

    multioutput_original = Feature(
        es["log"].ww["product_id"],
        parent_dataframe_name="customers",
        primitive=NMostCommon(n=2),
    )
    assert multioutput_original.get_name() == "N_MOST_COMMON(log.product_id, n=2)"

    featureslice_original = feature_base.FeatureOutputSlice(multioutput_original, 0)
    assert featureslice_original.get_name() == "N_MOST_COMMON(log.product_id, n=2)[0]"

    feature_type_list = [
        identity_original,
        agg_original,
        direct_original,
        transform_original,
        groupby_original,
        multioutput_original,
        featureslice_original,
    ]

    for feature_type in feature_type_list:
        serialize_name_unchanged(feature_type)


@pytest.fixture
def s3_client():
    _environ = os.environ.copy()
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.resource("s3")
        yield s3
    os.environ.clear()
    os.environ.update(_environ)


@pytest.fixture
def s3_bucket(s3_client, region="us-east-2"):
    location = {"LocationConstraint": region}
    s3_client.create_bucket(
        Bucket=BUCKET_NAME,
        ACL="public-read-write",
        CreateBucketConfiguration=location,
    )
    s3_bucket = s3_client.Bucket(BUCKET_NAME)
    yield s3_bucket


def test_serialize_features_mock_s3(es, s3_client, s3_bucket):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )

    save_features(features_original, TEST_S3_URL)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL="public-read-write")

    features_deserialized = load_features(TEST_S3_URL)
    assert_features(features_original, features_deserialized)


def test_serialize_features_mock_anon_s3(es, s3_client, s3_bucket):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )

    save_features(features_original, TEST_S3_URL, profile_name=False)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL="public-read-write")

    features_deserialized = load_features(TEST_S3_URL, profile_name=False)
    assert_features(features_original, features_deserialized)


@pytest.mark.parametrize("profile_name", ["test", False])
def test_s3_test_profile(es, s3_client, s3_bucket, setup_test_profile, profile_name):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )

    save_features(features_original, TEST_S3_URL, profile_name="test")

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL="public-read-write")

    features_deserialized = load_features(TEST_S3_URL, profile_name=profile_name)
    assert_features(features_original, features_deserialized)


@pytest.mark.parametrize("url,profile_name", [(S3_URL, False), (URL, None)])
def test_deserialize_features_s3(pd_es, url, profile_name):
    agg_primitives = [
        Sum,
        Std,
        Max,
        Skew,
        Min,
        Mean,
        Count,
        PercentTrue,
        NumUnique,
        Mode,
    ]

    trans_primitives = [Day, Year, Month, Weekday, Haversine, NumWords, NumCharacters]

    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=pd_es,
        features_only=True,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
    )

    features_deserialized = load_features(url, profile_name=profile_name)
    assert_features(features_original, features_deserialized)


def test_serialize_url(es):
    features_original = dfs(
        target_dataframe_name="sessions",
        entityset=es,
        features_only=True,
    )
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        save_features(features_original, URL)


def test_custom_feature_names_retained_during_serialization(pd_es, tmp_path):
    class MultiCumulative(TransformPrimitive):
        name = "multi_cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

    multi_output_trans_feat = Feature(
        pd_es["log"].ww["value"],
        primitive=MultiCumulative,
    )
    groupby_trans_feat = GroupByTransformFeature(
        pd_es["log"].ww["value"],
        primitive=MultiCumulative,
        groupby=pd_es["log"].ww["product_id"],
    )
    multi_output_agg_feat = Feature(
        pd_es["log"].ww["product_id"],
        parent_dataframe_name="customers",
        primitive=NMostCommon(n=2),
    )
    slice = FeatureOutputSlice(multi_output_trans_feat, 1)
    stacked_feat = Feature(slice, primitive=Negate)

    trans_names = ["cumulative_sum", "cumulative_max", "cumulative_min"]
    multi_output_trans_feat.set_feature_names(trans_names)
    groupby_trans_names = ["grouped_sum", "grouped_max", "grouped_min"]
    groupby_trans_feat.set_feature_names(groupby_trans_names)
    agg_names = ["first_most_common", "second_most_common"]
    multi_output_agg_feat.set_feature_names(agg_names)

    features = [
        multi_output_trans_feat,
        multi_output_agg_feat,
        groupby_trans_feat,
        stacked_feat,
    ]
    file = os.path.join(tmp_path, "features.json")
    save_features(features, file)
    deserialized_features = load_features(file)

    new_trans, new_agg, new_groupby, new_stacked = deserialized_features
    assert new_trans.get_feature_names() == trans_names
    assert new_agg.get_feature_names() == agg_names
    assert new_groupby.get_feature_names() == groupby_trans_names
    assert new_stacked.get_feature_names() == ["-(cumulative_max)"]


def test_deserializer_uses_common_primitive_instances_no_args(es, tmp_path):
    features = dfs(
        entityset=es,
        target_dataframe_name="products",
        features_only=True,
        agg_primitives=["sum"],
        trans_primitives=["is_null"],
    )

    is_null_features = [f for f in features if f.primitive.name == "is_null"]
    sum_features = [f for f in features if f.primitive.name == "sum"]

    # Make sure we have multiple features of each type
    assert len(is_null_features) > 1
    assert len(sum_features) > 1

    # DFS should use the same primitive instance for all features that share a primitive
    is_null_primitive = is_null_features[0].primitive
    sum_primitive = sum_features[0].primitive
    assert all([f.primitive is is_null_primitive for f in is_null_features])
    assert all([f.primitive is sum_primitive for f in sum_features])

    file = os.path.join(tmp_path, "features.json")
    save_features(features, file)
    deserialized_features = load_features(file)
    new_is_null_features = [
        f for f in deserialized_features if f.primitive.name == "is_null"
    ]
    new_sum_features = [f for f in deserialized_features if f.primitive.name == "sum"]

    # After deserialization all features that share a primitive should use the same primitive instance
    new_is_null_primitive = new_is_null_features[0].primitive
    new_sum_primitive = new_sum_features[0].primitive
    assert all([f.primitive is new_is_null_primitive for f in new_is_null_features])
    assert all([f.primitive is new_sum_primitive for f in new_sum_features])


def test_deserializer_uses_common_primitive_instances_with_args(es, tmp_path):
    # Single argument
    scalar1 = MultiplyNumericScalar(value=1)
    scalar5 = MultiplyNumericScalar(value=5)
    features = dfs(
        entityset=es,
        target_dataframe_name="products",
        features_only=True,
        agg_primitives=["sum"],
        trans_primitives=[scalar1, scalar5],
    )

    scalar1_features = [
        f
        for f in features
        if f.primitive.name == "multiply_numeric_scalar" and " * 1" in f.get_name()
    ]
    scalar5_features = [
        f
        for f in features
        if f.primitive.name == "multiply_numeric_scalar" and " * 5" in f.get_name()
    ]

    # Make sure we have multiple features of each type
    assert len(scalar1_features) > 1
    assert len(scalar5_features) > 1

    # DFS should use the the passed in primitive instance for all features
    assert all([f.primitive is scalar1 for f in scalar1_features])
    assert all([f.primitive is scalar5 for f in scalar5_features])

    file = os.path.join(tmp_path, "features.json")
    save_features(features, file)
    deserialized_features = load_features(file)

    new_scalar1_features = [
        f
        for f in deserialized_features
        if f.primitive.name == "multiply_numeric_scalar" and " * 1" in f.get_name()
    ]
    new_scalar5_features = [
        f
        for f in deserialized_features
        if f.primitive.name == "multiply_numeric_scalar" and " * 5" in f.get_name()
    ]

    # After deserialization all features that share a primitive should use the same primitive instance
    new_scalar1_primitive = new_scalar1_features[0].primitive
    new_scalar5_primitive = new_scalar5_features[0].primitive
    assert all([f.primitive is new_scalar1_primitive for f in new_scalar1_features])
    assert all([f.primitive is new_scalar5_primitive for f in new_scalar5_features])
    assert new_scalar1_primitive.value == 1
    assert new_scalar5_primitive.value == 5

    # Test primitive with multiple args - pandas only due to primitive compatibility
    if es.dataframe_type == Library.PANDAS:
        distance_to_holiday = DistanceToHoliday(
            holiday="Canada Day",
            country="Canada",
        )
        features = dfs(
            entityset=es,
            target_dataframe_name="customers",
            features_only=True,
            agg_primitives=[],
            trans_primitives=[distance_to_holiday],
        )

        distance_features = [
            f for f in features if f.primitive.name == "distance_to_holiday"
        ]

        assert len(distance_features) > 1

        # DFS should use the the passed in primitive instance for all features
        assert all([f.primitive is distance_to_holiday for f in distance_features])

        file = os.path.join(tmp_path, "distance_features.json")
        save_features(distance_features, file)
        new_distance_features = load_features(file)

        # After deserialization all features that share a primitive should use the same primitive instance
        new_distance_primitive = new_distance_features[0].primitive
        assert all(
            [f.primitive is new_distance_primitive for f in new_distance_features],
        )
        assert new_distance_primitive.holiday == "Canada Day"
        assert new_distance_primitive.country == "Canada"

    # Test primitive with list arg
    is_in = IsIn(list_of_outputs=[5, True, "coke zero"])
    features = dfs(
        entityset=es,
        target_dataframe_name="customers",
        features_only=True,
        agg_primitives=[],
        trans_primitives=[is_in],
    )

    is_in_features = [f for f in features if f.primitive.name == "isin"]
    assert len(is_in_features) > 1

    # DFS should use the the passed in primitive instance for all features
    assert all([f.primitive is is_in for f in is_in_features])

    file = os.path.join(tmp_path, "distance_features.json")
    save_features(is_in_features, file)
    new_is_in_features = load_features(file)

    # After deserialization all features that share a primitive should use the same primitive instance
    new_is_in_primitive = new_is_in_features[0].primitive
    assert all([f.primitive is new_is_in_primitive for f in new_is_in_features])
    assert new_is_in_primitive.list_of_outputs == [5, True, "coke zero"]


def test_can_serialize_word_set_for_number_of_common_words_feature(pd_es):
    # The word_set argument is passed in as a set, which is not JSON-serializable.
    # This test checks internal logic that converts the set to a list so it can be serialized
    common_word_set = {"hello", "my"}
    df = pd.DataFrame({"text": ["hello my name is hi"]})
    es = EntitySet()
    es.add_dataframe(dataframe_name="df", index="idx", dataframe=df, make_index=True)

    num_common_words = NumberOfCommonWords(word_set=common_word_set)
    fm, fd = dfs(
        entityset=es,
        target_dataframe_name="df",
        trans_primitives=[num_common_words],
    )

    feat = fd[-1]
    save_features([feat])
