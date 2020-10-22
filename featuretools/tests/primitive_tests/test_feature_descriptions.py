import os

from featuretools import describe_feature
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from featuretools.primitives import (
    Absolute,
    CumMean,
    EqualScalar,
    Mean,
    Mode,
    NMostCommon,
    NumUnique,
    Sum
)


def test_identity_description(es):
    feature = IdentityFeature(es['log']['session_id'])
    description = 'The "session_id".'

    assert describe_feature(feature) == description


def test_direct_description(es):
    feature = DirectFeature(es['customers']['loves_ice_cream'], es['sessions'])
    description = 'The "loves_ice_cream" of the instance of "customers" associated ' \
                  'with this instance of "sessions".'
    assert describe_feature(feature) == description

    deep_direct = DirectFeature(feature, es['log'])
    deep_description = 'The "loves_ice_cream" of the instance of "customers" ' \
                       'associated with the instance of "sessions" associated with ' \
                       'this instance of "log".'
    assert describe_feature(deep_direct) == deep_description


def test_transform_description(es):
    feature = TransformFeature(es['log']['value'], Absolute)
    description = 'The absolute value of the "value".'

    assert describe_feature(feature) == description


def test_groupby_transform_description(es):
    feature = GroupByTransformFeature(es['log']['value'], CumMean, es['log']['session_id'])
    description = 'The cumulative mean of the "value" for each "session_id".'

    assert describe_feature(feature) == description


def test_aggregation_description(es):
    feature = AggregationFeature(es['log']['value'], es['sessions'], Mean)
    description = 'The average of the "value" of all instances of "log" for each "id" in "sessions".'
    assert describe_feature(feature) == description

    stacked_agg = AggregationFeature(feature, es['customers'], Sum)
    stacked_description = 'The sum of "MEAN(log.value)" of all instances ' \
                          'of "sessions" for each "id" in "customers". "MEAN(log.value)" is t' + description[1:]
    assert describe_feature(stacked_agg) == stacked_description


def test_aggregation_description_where(es):
    where_feature = TransformFeature(es['log']['countrycode'], EqualScalar('US'))
    feature = AggregationFeature(es['log']['value'], es['sessions'],
                                 Mean, where=where_feature)
    description = 'The average of the "value" of all instances of "log" where the ' \
                  '"countrycode" is US for each "id" in "sessions".'

    assert describe_feature(feature) == description


def test_aggregation_description_use_previous(es):
    feature = AggregationFeature(es['log']['value'], es['sessions'],
                                 Mean, use_previous='5d')
    description = 'The average of the "value" of the previous 5 days of "log" for each "id" in "sessions".'

    assert describe_feature(feature) == description


def test_multi_output_description(es):
    feature = AggregationFeature(es['log']['zipcode'], es['sessions'], NMostCommon)
    first_slice = feature[0]
    second_slice = feature[1]

    description = 'The 3 most common values of the "zipcode" of all instances of "log" for each "id" in "sessions".'
    first_description = 'The most common value of the "zipcode" of all instances of "log" ' \
                        'for each "id" in "sessions".'
    second_description = 'The 2nd most common value of the "zipcode" of all instances of ' \
                         '"log" for each "id" in "sessions".'

    assert describe_feature(feature) == description
    assert describe_feature(first_slice) == first_description
    assert describe_feature(second_slice) == second_description


def test_metadata(es):
    identity_feature_descriptions = {'sessions: device_name': 'the name of the device used for each session'}
    agg_feat = AggregationFeature(es['sessions']['device_name'], es['customers'], NumUnique)
    agg_description = 'The number of unique elements in the name of the device used for each '\
                      'session of all instances of "sessions" for each "id" in "customers".'
    assert describe_feature(agg_feat, feature_descriptions=identity_feature_descriptions) == agg_description

    transform_feat = GroupByTransformFeature(es['log']['value'], CumMean, es['log']['session_id'])
    transform_description = 'The running average of the "value" for each "session_id".'
    primitive_templates = {"cum_mean": "the running average of {}"}
    assert describe_feature(transform_feat, primitive_templates=primitive_templates) == transform_description

    custom_agg = AggregationFeature(es['log']['zipcode'], es['customers'], Mode)
    auto_description = 'The most frequently occurring value of the "zipcode" of all instances of "log" for each "id" in "customers".'
    custom_agg_description = "the most frequently used zipcode"
    custom_feature_description = custom_agg_description[0].upper() + custom_agg_description[1:] + '.'
    feature_description_dict = {'customers: MODE(log.zipcode)': custom_agg_description}
    assert describe_feature(custom_agg) == auto_description
    assert describe_feature(custom_agg, feature_descriptions=feature_description_dict) == custom_feature_description

    this_directory = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(this_directory, 'description_metadata.json')
    assert describe_feature(agg_feat, metadata_file=metadata_path) == agg_description
    assert describe_feature(transform_feat, metadata_file=metadata_path) == transform_description
    assert describe_feature(custom_agg, metadata_file=metadata_path) == custom_feature_description
