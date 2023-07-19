import copy
import re

import pandas as pd
import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools import EntitySet, Feature, GroupByTransformFeature
from featuretools.entityset.entityset import LTI_COLUMN_NAME
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_base.utils import is_valid_input
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    Count,
    CumCount,
    CumMean,
    CumMin,
    CumSum,
    Day,
    Diff,
    Equal,
    Hour,
    IsIn,
    IsNull,
    Last,
    Mean,
    Mode,
    Month,
    Negate,
    NMostCommon,
    Not,
    NotEqual,
    NumCharacters,
    NumTrue,
    NumUnique,
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingOutlierCount,
    RollingSTD,
    Sum,
    TimeSincePrevious,
    TransformPrimitive,
    Trend,
    Year,
)
from featuretools.synthesis import DeepFeatureSynthesis
from featuretools.tests.testing_utils import (
    feature_with_name,
    make_ecommerce_entityset,
    number_of_features_with_name_like,
)
from featuretools.utils.gen_utils import Library


def test_makes_agg_features_from_str(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=["sum"],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "SUM(log.value)")


def test_makes_agg_features_from_mixed_str(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Count, "sum"],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "SUM(log.value)")
    assert feature_with_name(features, "COUNT(log)")


def test_makes_agg_features(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "SUM(log.value)")


def test_only_makes_supplied_agg_feat(es):
    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        max_depth=3,
    )
    dfs_obj = DeepFeatureSynthesis(agg_primitives=[Sum], **kwargs)

    features = dfs_obj.build_features()

    def find_other_agg_features(features):
        return [
            f
            for f in features
            if (isinstance(f, AggregationFeature) and not isinstance(f.primitive, Sum))
            or len(
                [
                    g
                    for g in f.base_features
                    if isinstance(g, AggregationFeature)
                    and not isinstance(g.primitive, Sum)
                ],
            )
            > 0
        ]

    other_agg_features = find_other_agg_features(features)
    assert len(other_agg_features) == 0


def test_errors_unsupported_primitives(es):
    bad_trans_prim = CumSum()
    bad_agg_prim = NumUnique()
    bad_trans_prim.compatibility, bad_agg_prim.compatibility = [], []
    library = es.dataframe_type
    error_text = "Selected primitives are incompatible with {} EntitySets: cum_sum, num_unique".format(
        library.value,
    )
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=es,
            agg_primitives=[bad_agg_prim],
            trans_primitives=[bad_trans_prim],
        )


def test_errors_unsupported_primitives_spark(spark_es):
    bad_trans_prim = CumSum()
    bad_agg_prim = NumUnique()
    bad_trans_prim.spark_compatible, bad_agg_prim.spark_compatible = False, False
    error_text = "Selected primitives are incompatible with Spark EntitySets: cum_sum"
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=spark_es,
            agg_primitives=[bad_agg_prim],
            trans_primitives=[bad_trans_prim],
        )


def test_error_for_missing_target_dataframe(es):
    error_text = (
        "Provided target dataframe missing_dataframe does not exist in ecommerce"
    )
    with pytest.raises(KeyError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="missing_dataframe",
            entityset=es,
            agg_primitives=[Last],
            trans_primitives=[],
            ignore_dataframes=["log"],
        )

    es_without_id = EntitySet()
    error_text = (
        "Provided target dataframe missing_dataframe does not exist in entity set"
    )
    with pytest.raises(KeyError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="missing_dataframe",
            entityset=es_without_id,
            agg_primitives=[Last],
            trans_primitives=[],
            ignore_dataframes=["log"],
        )


def test_ignores_dataframes(es):
    error_text = "ignore_dataframes must be a list"
    with pytest.raises(TypeError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=es,
            agg_primitives=[Sum],
            trans_primitives=[],
            ignore_dataframes="log",
        )

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        ignore_dataframes=["log"],
    )

    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        dataframes = [d.dataframe_name for d in deps]
        assert "log" not in dataframes


def test_ignores_columns(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        ignore_columns={"log": ["value"]},
    )
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        identities = [d for d in deps if isinstance(d, IdentityFeature)]
        columns = [d.column_name for d in identities if d.dataframe_name == "log"]
        assert "value" not in columns


def test_ignore_columns_input_type(es):
    error_msg = r"ignore_columns should be dict\[str -> list\]"  # need to use string literals to avoid regex params
    wrong_input_type = {"log": "value"}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_dataframe_name="log",
            entityset=es,
            ignore_columns=wrong_input_type,
        )


def test_ignore_columns_with_nonstring_values(es):
    error_msg = "list in ignore_columns must only have string values"
    wrong_input_list = {"log": ["a", "b", 3]}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_dataframe_name="log",
            entityset=es,
            ignore_columns=wrong_input_list,
        )


def test_ignore_columns_with_nonstring_keys(es):
    error_msg = r"ignore_columns should be dict\[str -> list\]"  # need to use string literals to avoid regex params
    wrong_input_keys = {1: ["a", "b", "c"]}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_dataframe_name="log",
            entityset=es,
            ignore_columns=wrong_input_keys,
        )


def test_makes_dfeatures(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "customers.age")


def test_makes_trans_feat(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[Hour],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "HOUR(datetime)")


def test_handles_diff_dataframe_groupby(pd_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        groupby_trans_primitives=[Diff],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "DIFF(value) by session_id")
    assert feature_with_name(features, "DIFF(value) by product_id")


def test_handles_time_since_previous_dataframe_groupby(pd_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        groupby_trans_primitives=[TimeSincePrevious],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "TIME_SINCE_PREVIOUS(datetime) by session_id")


# M TODO
# def test_handles_cumsum_dataframe_groupby(pd_es):
#     dfs_obj = DeepFeatureSynthesis(target_dataframe_name='sessions',
#                                    entityset=pd_es,
#                                    agg_primitives=[],
#                                    trans_primitives=[CumMean])

#     features = dfs_obj.build_features()
#     assert (feature_with_name(features, u'customers.CUM_MEAN(age by région_id)'))


def test_only_makes_supplied_trans_feat(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[Hour],
    )

    features = dfs_obj.build_features()
    other_trans_features = [
        f
        for f in features
        if (isinstance(f, TransformFeature) and not isinstance(f.primitive, Hour))
        or len(
            [
                g
                for g in f.base_features
                if isinstance(g, TransformFeature) and not isinstance(g.primitive, Hour)
            ],
        )
        > 0
    ]
    assert len(other_trans_features) == 0


def test_makes_dfeatures_of_agg_primitives(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=["max"],
        trans_primitives=[],
    )
    features = dfs_obj.build_features()

    assert feature_with_name(features, "customers.MAX(log.value)")


def test_makes_agg_features_of_trans_primitives(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Mean],
        trans_primitives=[NumCharacters],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "MEAN(log.NUM_CHARACTERS(comments))")


def test_makes_agg_features_with_where(es):
    # TODO: Update to work with Dask and Spark `es` fixture when issue #978 is closed
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support add_interesting_values")
    es.add_interesting_values()

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Count],
        where_primitives=[Count],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "COUNT(log WHERE priority_level = 0)")

    # make sure they are made using direct features too
    assert feature_with_name(features, "COUNT(log WHERE products.department = food)")


def test_make_groupby_features(pd_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
    )
    features = dfs_obj.build_features()
    assert feature_with_name(features, "CUM_SUM(value) by session_id")


def test_make_indirect_groupby_features(pd_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
    )
    features = dfs_obj.build_features()
    assert feature_with_name(features, "CUM_SUM(products.rating) by session_id")


def test_make_groupby_features_with_id(pd_es):
    # Need to convert customer_id to categorical column in order to build desired feature
    pd_es["sessions"].ww.set_types(
        logical_types={"customer_id": "Categorical"},
        semantic_tags={"customer_id": "foreign_key"},
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_count"],
    )
    features = dfs_obj.build_features()

    assert feature_with_name(features, "CUM_COUNT(customer_id) by customer_id")


def test_make_groupby_features_with_diff_id(pd_es):
    # Need to convert cohort to categorical column in order to build desired feature
    pd_es["customers"].ww.set_types(
        logical_types={"cohort": "Categorical"},
        semantic_tags={"cohort": "foreign_key"},
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_count"],
    )
    features = dfs_obj.build_features()

    groupby_with_diff_id = "CUM_COUNT(cohort) by région_id"
    assert feature_with_name(features, groupby_with_diff_id)


def test_make_groupby_features_with_agg(pd_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="cohorts",
        entityset=pd_es,
        agg_primitives=["sum"],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
    )
    features = dfs_obj.build_features()
    agg_on_groupby_name = "SUM(customers.CUM_SUM(age) by région_id)"
    assert feature_with_name(features, agg_on_groupby_name)


def test_bad_groupby_feature(es):
    msg = re.escape(
        "Unknown groupby transform primitive max. "
        "Call ft.primitives.list_primitives() to get "
        "a list of available primitives",
    )
    with pytest.raises(ValueError, match=msg):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["sum"],
            trans_primitives=[],
            groupby_trans_primitives=["Max"],
        )


@pytest.mark.parametrize(
    "rolling_primitive",
    [
        RollingMax,
        RollingMean,
        RollingMin,
        RollingOutlierCount,
        RollingSTD,
    ],
)
@pytest.mark.parametrize(
    "window_length, gap",
    [
        (7, 3),
        ("7d", "3d"),
    ],
)
def test_make_rolling_features(window_length, gap, rolling_primitive, pd_es):
    rolling_primitive_obj = rolling_primitive(
        window_length=window_length,
        gap=gap,
        min_periods=5,
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[rolling_primitive_obj],
    )
    features = dfs_obj.build_features()
    rolling_transform_name = f"{rolling_primitive.name.upper()}(datetime, value_many_nans, window_length={window_length}, gap={gap}, min_periods=5)"
    assert feature_with_name(features, rolling_transform_name)


@pytest.mark.parametrize(
    "window_length, gap",
    [
        (7, 3),
        ("7d", "3d"),
    ],
)
def test_make_rolling_count_off_datetime_feature(window_length, gap, pd_es):
    rolling_count = RollingCount(window_length=window_length, min_periods=gap)
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[rolling_count],
    )
    features = dfs_obj.build_features()
    rolling_transform_name = (
        f"ROLLING_COUNT(datetime, window_length={window_length}, min_periods={gap})"
    )
    assert feature_with_name(features, rolling_transform_name)


def test_abides_by_max_depth_param(es):
    for i in [0, 1, 2, 3]:
        dfs_obj = DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=es,
            agg_primitives=[Sum],
            trans_primitives=[],
            max_depth=i,
        )

        features = dfs_obj.build_features()
        for f in features:
            assert f.get_depth() <= i


def test_max_depth_single_table(transform_es):
    assert len(transform_es.dataframe_dict) == 1

    def make_dfs_obj(max_depth):
        dfs_obj = DeepFeatureSynthesis(
            target_dataframe_name="first",
            entityset=transform_es,
            trans_primitives=[AddNumeric],
            max_depth=max_depth,
        )
        return dfs_obj

    for i in [-1, 0, 1, 2]:
        if i in [-1, 2]:
            match = (
                "Only one dataframe in entityset, changing max_depth to 1 "
                "since deeper features cannot be created"
            )
            with pytest.warns(UserWarning, match=match):
                dfs_obj = make_dfs_obj(i)
        else:
            dfs_obj = make_dfs_obj(i)

        features = dfs_obj.build_features()
        assert len(features) > 0
        if i != 0:
            # at least one depth 1 feature made
            assert any([f.get_depth() == 1 for f in features])
            # no depth 2 or higher even with max_depth=2
            assert all([f.get_depth() <= 1 for f in features])
        else:
            # no depth 1 or higher features with max_depth=0
            assert all([f.get_depth() == 0 for f in features])


def test_drop_contains(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=1,
        seed_features=[],
        drop_contains=[],
    )
    features = dfs_obj.build_features()
    to_drop = features[2]
    partial_name = to_drop.get_name()[:5]

    dfs_drop = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=1,
        seed_features=[],
        drop_contains=[partial_name],
    )
    features = dfs_drop.build_features()
    assert to_drop.get_name() not in [f.get_name() for f in features]


def test_drop_exact(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=1,
        seed_features=[],
        drop_exact=[],
    )
    features = dfs_obj.build_features()
    to_drop = features[2]
    name = to_drop.get_name()
    dfs_drop = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=1,
        seed_features=[],
        drop_exact=[name],
    )
    features = dfs_drop.build_features()
    assert name not in [f.get_name() for f in features]


def test_seed_features(es):
    seed_feature_sessions = (
        Feature(es["log"].ww["id"], parent_dataframe_name="sessions", primitive=Count)
        > 2
    )
    seed_feature_log = Feature(es["log"].ww["comments"], primitive=NumCharacters)
    session_agg = Feature(
        seed_feature_log,
        parent_dataframe_name="sessions",
        primitive=Mean,
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Mean],
        trans_primitives=[],
        max_depth=2,
        seed_features=[seed_feature_sessions, seed_feature_log],
    )
    features = dfs_obj.build_features()
    assert seed_feature_sessions.get_name() in [f.get_name() for f in features]
    assert session_agg.get_name() in [f.get_name() for f in features]


def test_does_not_make_agg_of_direct_of_target_dataframe(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support the Last primitive")

    count_sessions = Feature(
        es["sessions"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Last],
        trans_primitives=[],
        max_depth=2,
        seed_features=[count_sessions],
    )
    features = dfs_obj.build_features()
    # this feature is meaningless because customers.COUNT(sessions) is already defined on
    # the customers dataframe
    assert not feature_with_name(features, "LAST(sessions.customers.COUNT(sessions))")
    assert not feature_with_name(features, "LAST(sessions.customers.age)")


def test_dfs_builds_on_seed_features_more_than_max_depth(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support the Last and Mode primitives")

    seed_feature_sessions = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    seed_feature_log = Feature(es["log"].ww["datetime"], primitive=Hour)
    session_agg = Feature(
        seed_feature_log,
        parent_dataframe_name="sessions",
        primitive=Last,
    )

    # Depth of this feat is 2 relative to session_agg, the seed feature,
    # which is greater than max_depth so it shouldn't be built
    session_agg_trans = DirectFeature(
        Feature(session_agg, parent_dataframe_name="customers", primitive=Mode),
        "sessions",
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Last, Count],
        trans_primitives=[],
        max_depth=1,
        seed_features=[seed_feature_sessions, seed_feature_log],
    )
    features = dfs_obj.build_features()
    assert seed_feature_sessions.get_name() in [f.get_name() for f in features]
    assert session_agg.get_name() in [f.get_name() for f in features]
    assert session_agg_trans.get_name() not in [f.get_name() for f in features]


def test_dfs_includes_seed_features_greater_than_max_depth(es):
    session_agg = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Sum,
    )
    customer_agg = Feature(
        session_agg,
        parent_dataframe_name="customers",
        primitive=Mean,
    )
    assert customer_agg.get_depth() == 2

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Mean],
        trans_primitives=[],
        max_depth=1,
        seed_features=[customer_agg],
    )
    features = dfs_obj.build_features()
    assert feature_with_name(features=features, name=customer_agg.get_name())


def test_allowed_paths(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support the Last primitive")

    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Last],
        trans_primitives=[],
        max_depth=2,
        seed_features=[],
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features_unconstrained = dfs_unconstrained.build_features()

    unconstrained_names = [f.get_name() for f in features_unconstrained]
    customers_session_feat = Feature(
        es["sessions"].ww["device_type"],
        parent_dataframe_name="customers",
        primitive=Last,
    )
    customers_session_log_feat = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Last,
    )
    assert customers_session_feat.get_name() in unconstrained_names
    assert customers_session_log_feat.get_name() in unconstrained_names

    dfs_constrained = DeepFeatureSynthesis(
        allowed_paths=[["customers", "sessions"]], **kwargs
    )
    features = dfs_constrained.build_features()
    names = [f.get_name() for f in features]
    assert customers_session_feat.get_name() in names
    assert customers_session_log_feat.get_name() not in names


def test_max_features(es):
    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=2,
        seed_features=[],
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features_unconstrained = dfs_unconstrained.build_features()
    dfs_unconstrained_with_arg = DeepFeatureSynthesis(max_features=-1, **kwargs)
    feats_unconstrained_with_arg = dfs_unconstrained_with_arg.build_features()
    dfs_constrained = DeepFeatureSynthesis(max_features=1, **kwargs)
    features = dfs_constrained.build_features()
    assert len(features_unconstrained) == len(feats_unconstrained_with_arg)
    assert len(features) == 1


def test_where_primitives(es):
    es.add_interesting_values(dataframe_name="sessions", values={"device_type": [0]})
    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Count, Sum],
        trans_primitives=[Absolute],
        max_depth=3,
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    dfs_constrained = DeepFeatureSynthesis(where_primitives=["sum"], **kwargs)
    features_unconstrained = dfs_unconstrained.build_features()
    features = dfs_constrained.build_features()

    where_feats_unconstrained = [
        f
        for f in features_unconstrained
        if isinstance(f, AggregationFeature) and f.where is not None
    ]
    where_feats = [
        f for f in features if isinstance(f, AggregationFeature) and f.where is not None
    ]

    assert len(where_feats_unconstrained) >= 1

    assert (
        len([f for f in where_feats_unconstrained if isinstance(f.primitive, Sum)]) == 0
    )
    assert (
        len([f for f in where_feats_unconstrained if isinstance(f.primitive, Count)])
        > 0
    )

    assert len([f for f in where_feats if isinstance(f.primitive, Sum)]) > 0
    assert len([f for f in where_feats if isinstance(f.primitive, Count)]) == 0
    assert (
        len(
            [
                d
                for f in where_feats
                for d in f.get_dependencies(deep=True)
                if isinstance(d.primitive, Absolute)
            ],
        )
        > 0
    )


def test_stacking_where_primitives(es):
    # TODO: Update to work with Dask supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask and Spark EntitySets do not support the Last primitive")
    es = copy.deepcopy(es)
    es.add_interesting_values(dataframe_name="sessions", values={"device_type": [0]})
    es.add_interesting_values(
        dataframe_name="log",
        values={"product_id": ["coke_zero"]},
    )
    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Count, Last],
        max_depth=3,
    )
    dfs_where_stack_limit_1 = DeepFeatureSynthesis(
        where_primitives=["last", Count], **kwargs
    )
    dfs_where_stack_limit_2 = DeepFeatureSynthesis(
        where_primitives=["last", Count], where_stacking_limit=2, **kwargs
    )
    stack_limit_1_features = dfs_where_stack_limit_1.build_features()
    stack_limit_2_features = dfs_where_stack_limit_2.build_features()

    where_stack_1_feats = [
        f
        for f in stack_limit_1_features
        if isinstance(f, AggregationFeature) and f.where is not None
    ]
    where_stack_2_feats = [
        f
        for f in stack_limit_2_features
        if isinstance(f, AggregationFeature) and f.where is not None
    ]

    assert len(where_stack_1_feats) >= 1
    assert len(where_stack_2_feats) >= 1

    assert len([f for f in where_stack_1_feats if isinstance(f.primitive, Last)]) > 0
    assert len([f for f in where_stack_1_feats if isinstance(f.primitive, Count)]) > 0

    assert len([f for f in where_stack_2_feats if isinstance(f.primitive, Last)]) > 0
    assert len([f for f in where_stack_2_feats if isinstance(f.primitive, Count)]) > 0

    stacked_where_limit_1_feats = []
    stacked_where_limit_2_feats = []
    where_double_where_tuples = [
        (where_stack_1_feats, stacked_where_limit_1_feats),
        (where_stack_2_feats, stacked_where_limit_2_feats),
    ]
    for where_list, double_where_list in where_double_where_tuples:
        for feature in where_list:
            for base_feat in feature.base_features:
                if (
                    isinstance(base_feat, AggregationFeature)
                    and base_feat.where is not None
                ):
                    double_where_list.append(feature)

    assert len(stacked_where_limit_1_feats) == 0
    assert len(stacked_where_limit_2_feats) > 0


def test_where_different_base_feats(es):
    es.add_interesting_values(dataframe_name="sessions", values={"device_type": [0]})

    kwargs = dict(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[Sum, Count],
        where_primitives=[Sum, Count],
        max_depth=3,
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features = dfs_unconstrained.build_features()
    where_feats = [
        f.unique_name()
        for f in features
        if isinstance(f, AggregationFeature) and f.where is not None
    ]
    not_where_feats = [
        f.unique_name()
        for f in features
        if isinstance(f, AggregationFeature) and f.where is None
    ]
    for name in not_where_feats:
        assert name not in where_feats


def test_dfeats_where(es):
    # TODO: Update to work with Dask `es` fixture when issue #978 is closed
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask and Spark EntitySets do not support add_interesting_values")
    es.add_interesting_values()

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Count],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()

    # test to make sure we build direct features of agg features with where clause
    assert feature_with_name(features, "customers.COUNT(log WHERE priority_level = 0)")

    assert feature_with_name(
        features,
        "COUNT(log WHERE products.department = electronics)",
    )


def test_commutative(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[AddNumeric],
        max_depth=3,
    )
    feats = dfs_obj.build_features()

    add_feats = [f for f in feats if isinstance(f.primitive, AddNumeric)]

    # Check that there are no two AddNumeric features with the same base
    # features.
    unordered_args = set()
    for f in add_feats:
        arg1, arg2 = f.base_features
        args_set = frozenset({arg1.unique_name(), arg2.unique_name()})
        unordered_args.add(args_set)

    assert len(add_feats) == len(unordered_args)


def test_transform_consistency(transform_es):
    # Generate features
    transform_es["first"].ww.set_types(
        logical_types={"b": "BooleanNullable", "b1": "BooleanNullable"},
    )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="first",
        entityset=transform_es,
        trans_primitives=["and", "add_numeric", "or"],
        max_depth=1,
    )
    feature_defs = dfs_obj.build_features()

    # Check for correct ordering of features
    assert feature_with_name(feature_defs, "a")
    assert feature_with_name(feature_defs, "b")
    assert feature_with_name(feature_defs, "b1")
    assert feature_with_name(feature_defs, "b12")
    assert feature_with_name(feature_defs, "P")

    assert feature_with_name(feature_defs, "AND(b, b1)")
    assert not feature_with_name(
        feature_defs,
        "AND(b1, b)",
    )  # make sure it doesn't exist the other way
    assert feature_with_name(feature_defs, "a + P")
    assert feature_with_name(feature_defs, "b12 + P")
    assert feature_with_name(feature_defs, "a + b12")
    assert feature_with_name(feature_defs, "OR(b, b1)")


def test_transform_no_stack_agg(es):
    # TODO: Update to work with Dask and Spark supported primitives
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support the NMostCommon primitive")
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[NMostCommon],
        trans_primitives=[NotEqual],
        max_depth=3,
    )
    feature_defs = dfs_obj.build_features()

    assert not feature_with_name(
        feature_defs,
        "id != N_MOST_COMMON(sessions.device_type)",
    )


def test_initialized_trans_prim(es):
    prim = IsIn(list_of_outputs=["coke zero"])
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[prim],
    )

    features = dfs_obj.build_features()

    assert feature_with_name(features, "product_id.isin(['coke zero'])")


def test_initialized_agg_prim(es):
    # TODO: Update to work with Dask and Spark supported primitives
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask EntitySets do not support the NMostCommon primitive")
    ThreeMost = NMostCommon(n=3)
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[ThreeMost],
        trans_primitives=[],
    )
    features = dfs_obj.build_features()

    assert feature_with_name(features, "N_MOST_COMMON(log.subregioncode)")


def test_return_types(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Dask and Spark EntitySets do not support the NMostCommon primitive",
        )
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Count, NMostCommon],
        trans_primitives=[Absolute, Hour, IsIn],
    )

    discrete = ColumnSchema(semantic_tags={"category"})
    numeric = ColumnSchema(semantic_tags={"numeric"})
    datetime = ColumnSchema(logical_type=Datetime)

    f1 = dfs_obj.build_features(return_types=None)
    f2 = dfs_obj.build_features(return_types=[discrete])
    f3 = dfs_obj.build_features(return_types="all")
    f4 = dfs_obj.build_features(return_types=[datetime])

    f1_types = [f.column_schema for f in f1]
    f2_types = [f.column_schema for f in f2]
    f3_types = [f.column_schema for f in f3]
    f4_types = [f.column_schema for f in f4]

    assert any([is_valid_input(schema, discrete) for schema in f1_types])
    assert any([is_valid_input(schema, numeric) for schema in f1_types])
    assert not any([is_valid_input(schema, datetime) for schema in f1_types])

    assert any([is_valid_input(schema, discrete) for schema in f2_types])
    assert not any([is_valid_input(schema, numeric) for schema in f2_types])
    assert not any([is_valid_input(schema, datetime) for schema in f2_types])

    assert any([is_valid_input(schema, discrete) for schema in f3_types])
    assert any([is_valid_input(schema, numeric) for schema in f3_types])
    assert any([is_valid_input(schema, datetime) for schema in f3_types])

    assert not any([is_valid_input(schema, discrete) for schema in f4_types])
    assert not any([is_valid_input(schema, numeric) for schema in f4_types])
    assert any([is_valid_input(schema, datetime) for schema in f4_types])


def test_checks_primitives_correct_type(es):
    error_text = (
        "Primitive <class \\'featuretools\\.primitives\\.standard\\."
        "transform\\.datetime\\.hour\\.Hour\\'> in "
        "agg_primitives is not an aggregation primitive"
    )
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=es,
            agg_primitives=[Hour],
            trans_primitives=[],
        )

    error_text = (
        "Primitive <class \\'featuretools\\.primitives\\.standard\\."
        "aggregation\\.sum_primitive\\.Sum\\'> in trans_primitives "
        "is not a transform primitive"
    )
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="sessions",
            entityset=es,
            agg_primitives=[],
            trans_primitives=[Sum],
        )


def test_makes_agg_features_along_multiple_paths(diamond_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="regions",
        entityset=diamond_es,
        agg_primitives=["mean"],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "MEAN(customers.transactions.amount)")
    assert feature_with_name(features, "MEAN(stores.transactions.amount)")


def test_makes_direct_features_through_multiple_relationships(games_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="games",
        entityset=games_es,
        agg_primitives=["mean"],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()

    teams = ["home", "away"]
    for forward in teams:
        for backward in teams:
            for col in teams:
                f = "teams[%s_team_id].MEAN(games[%s_team_id].%s_team_score)" % (
                    forward,
                    backward,
                    col,
                )
                assert feature_with_name(features, f)


def test_stacks_multioutput_features(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Dask EntitySets do not support the NumUnique and NMostCommon primitives",
        )

    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [ColumnSchema(logical_type=Datetime)]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 6

        def get_function(self):
            def test_f(x):
                times = pd.Series(x)
                units = ["year", "month", "day", "hour", "minute", "second"]
                return [times.apply(lambda x: getattr(x, unit)) for unit in units]

            return test_f

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[NumUnique, NMostCommon(n=3)],
        trans_primitives=[TestTime, Diff],
        max_depth=4,
    )
    feat = dfs_obj.build_features()

    for i in range(3):
        f = "NUM_UNIQUE(sessions.N_MOST_COMMON(log.countrycode)[%d])" % i
        assert feature_with_name(feat, f)


def test_seed_multi_output_feature_stacking(es):
    # TODO: Update to work with Dask and Spark supported primitive
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Dask EntitySets do not support the NMostCommon and NumUnique primitives",
        )
    threecommon = NMostCommon(3)
    tc = Feature(
        es["log"].ww["product_id"],
        parent_dataframe_name="sessions",
        primitive=threecommon,
    )

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        seed_features=[tc],
        agg_primitives=[NumUnique],
        trans_primitives=[],
        max_depth=4,
    )
    feat = dfs_obj.build_features()

    for i in range(3):
        f = "NUM_UNIQUE(sessions.N_MOST_COMMON(log.product_id)[%d])" % i
        assert feature_with_name(feat, f)


def test_makes_direct_features_along_multiple_paths(diamond_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="transactions",
        entityset=diamond_es,
        max_depth=3,
        agg_primitives=[],
        trans_primitives=[],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "customers.regions.name")
    assert feature_with_name(features, "stores.regions.name")


def test_does_not_make_trans_of_single_direct_feature(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[],
        trans_primitives=["weekday"],
        max_depth=2,
    )

    features = dfs_obj.build_features()

    assert not feature_with_name(features, "WEEKDAY(customers.signup_date)")
    assert feature_with_name(features, "customers.WEEKDAY(signup_date)")


def test_makes_trans_of_multiple_direct_features(diamond_es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="transactions",
        entityset=diamond_es,
        agg_primitives=["mean"],
        trans_primitives=[Equal],
        max_depth=4,
    )

    features = dfs_obj.build_features()

    # Make trans of direct and non-direct
    assert feature_with_name(features, "amount = stores.MEAN(transactions.amount)")

    # Make trans of direct features on different dataframes
    assert feature_with_name(
        features,
        "customers.MEAN(transactions.amount) = stores.square_ft",
    )

    # Make trans of direct features on same dataframe with different paths.
    assert feature_with_name(features, "customers.regions.name = stores.regions.name")

    # Don't make trans of direct features with same path.
    assert not feature_with_name(
        features,
        "stores.square_ft = stores.MEAN(transactions.amount)",
    )
    assert not feature_with_name(
        features,
        "stores.MEAN(transactions.amount) = stores.square_ft",
    )

    # The naming of the below is confusing but this is a direct feature of a transform.
    assert feature_with_name(features, "stores.MEAN(transactions.amount) = square_ft")


def test_makes_direct_of_agg_of_trans_on_target(es):
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=["mean"],
        trans_primitives=[Absolute],
        max_depth=3,
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "sessions.MEAN(log.ABSOLUTE(value))")


def test_primitive_options_errors(es):
    wrong_key_options = {"mean": {"ignore_dataframe": ["sessions"]}}
    wrong_type_list = {"mean": {"ignore_dataframes": "sessions"}}
    wrong_type_dict = {"mean": {"ignore_columns": {"sessions": "product_id"}}}
    conflicting_primitive_options = {
        ("count", "mean"): {"ignore_dataframes": ["sessions"]},
        "mean": {"include_dataframes": ["sessions"]},
    }
    invalid_dataframe = {"mean": {"include_dataframes": ["invalid_dataframe"]}}
    invalid_column_dataframe = {
        "mean": {"include_columns": {"invalid_dataframe": ["product_id"]}},
    }
    invalid_column = {"mean": {"include_columns": {"sessions": ["invalid_column"]}}}
    key_error_text = "Unrecognized primitive option 'ignore_dataframe' for mean"
    list_error_text = "Incorrect type formatting for 'ignore_dataframes' for mean"
    dict_error_text = "Incorrect type formatting for 'ignore_columns' for mean"
    conflicting_error_text = "Multiple options found for primitive mean"
    invalid_dataframe_warning = "Dataframe 'invalid_dataframe' not in entityset"
    invalid_column_warning = "Column 'invalid_column' not in dataframe 'sessions'"
    with pytest.raises(KeyError, match=key_error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=wrong_key_options,
        )
    with pytest.raises(TypeError, match=list_error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=wrong_type_list,
        )
    with pytest.raises(TypeError, match=dict_error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=wrong_type_dict,
        )
    with pytest.raises(KeyError, match=conflicting_error_text):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=conflicting_primitive_options,
        )
    with pytest.warns(UserWarning, match=invalid_dataframe_warning) as record:
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=invalid_dataframe,
        )
    assert len(record) == 1
    with pytest.warns(UserWarning, match=invalid_dataframe_warning) as record:
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=invalid_column_dataframe,
        )
    assert len(record) == 1
    with pytest.warns(UserWarning, match=invalid_column_warning) as record:
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mean"],
            trans_primitives=[],
            primitive_options=invalid_column,
        )
    assert len(record) == 1


def test_primitive_options(es):
    options = {
        "sum": {"include_columns": {"customers": ["age"]}},
        "mean": {"include_dataframes": ["customers"]},
        "mode": {"ignore_dataframes": ["sessions"]},
        "num_unique": {"ignore_columns": {"customers": ["engagement_level"]}},
    }
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="cohorts",
        entityset=es,
        primitive_options=options,
    )
    features = dfs_obj.build_features()

    for f in features:
        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        columns = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Sum):
            for identity_base in columns:
                if identity_base.dataframe_name == "customers":
                    assert identity_base.get_name() == "age"
        if isinstance(f.primitive, Mean):
            assert all([df_name in ["customers"] for df_name in df_names])
        if isinstance(f.primitive, Mode):
            assert "sessions" not in df_names
        if isinstance(f.primitive, NumUnique):
            for identity_base in columns:
                assert not (
                    identity_base.dataframe_name == "customers"
                    and identity_base.get_name() == "engagement_level"
                )

    options = {
        "month": {"ignore_columns": {"customers": ["birthday"]}},
        "day": {"include_columns": {"customers": ["signup_date", "upgrade_date"]}},
        "num_characters": {"ignore_dataframes": ["customers"]},
        "year": {"include_dataframes": ["customers"]},
    }
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        agg_primitives=[],
        ignore_dataframes=["cohort"],
        primitive_options=options,
    )
    features = dfs_obj.build_features()
    assert not any([isinstance(f, NumCharacters) for f in features])
    for f in features:
        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        columns = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Month):
            for identity_base in columns:
                assert not (
                    identity_base.dataframe_name == "customers"
                    and identity_base.get_name() == "birthday"
                )
        if isinstance(f.primitive, Day):
            for identity_base in columns:
                if identity_base.dataframe_name == "customers":
                    assert (
                        identity_base.get_name() == "signup_date"
                        or identity_base.get_name() == "upgrade_date"
                    )
        if isinstance(f.primitive, Year):
            assert all([df_name in ["customers"] for df_name in df_names])


def test_primitive_options_with_globals(es):
    # non-overlapping ignore_dataframes
    options = {"mode": {"ignore_dataframes": ["sessions"]}}
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="cohorts",
        entityset=es,
        ignore_dataframes=["régions"],
        primitive_options=options,
    )
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        assert "régions" not in df_names
        if isinstance(f.primitive, Mode):
            assert "sessions" not in df_names

    # non-overlapping ignore_columns
    options = {"num_unique": {"ignore_columns": {"customers": ["engagement_level"]}}}
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        ignore_columns={"customers": ["région_id"]},
        primitive_options=options,
    )
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        columns = [d for d in deps if isinstance(d, IdentityFeature)]
        for identity_base in columns:
            assert not (
                identity_base.dataframe_name == "customers"
                and identity_base.get_name() == "région_id"
            )
        if isinstance(f.primitive, NumUnique):
            for identity_base in columns:
                assert not (
                    identity_base.dataframe_name == "customers"
                    and identity_base.get_name() == "engagement_level"
                )

    # Overlapping globals/options with ignore_dataframes
    options = {
        "mode": {
            "include_dataframes": ["sessions", "customers"],
            "ignore_columns": {"customers": ["région_id"]},
        },
        "num_unique": {
            "include_dataframes": ["sessions", "customers"],
            "include_columns": {"sessions": ["device_type"], "customers": ["age"]},
        },
        "month": {"ignore_columns": {"cohorts": ["cohort_end"]}},
    }
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="cohorts",
        entityset=es,
        ignore_dataframes=["sessions"],
        ignore_columns={"customers": ["age"]},
        primitive_options=options,
    )
    features = dfs_obj.build_features()
    for f in features:
        assert f.primitive.name != "month"
        # ignoring cohorts means no features are created
        assert not isinstance(f.primitive, Month)

        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        columns = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Mode):
            assert [all([df_name in ["sessions", "customers"] for df_name in df_names])]
            for identity_base in columns:
                assert not (
                    identity_base.dataframe_name == "customers"
                    and (
                        identity_base.get_name() == "age"
                        or identity_base.get_name() == "région_id"
                    )
                )
        elif isinstance(f.primitive, NumUnique):
            assert [all([df_name in ["sessions", "customers"] for df_name in df_names])]
            for identity_base in columns:
                if identity_base.dataframe_name == "sessions":
                    assert identity_base.get_name() == "device_type"
        # All other primitives ignore 'sessions' and 'age'
        else:
            assert "sessions" not in df_names
            for identity_base in columns:
                assert not (
                    identity_base.dataframe_name == "customers"
                    and identity_base.get_name() == "age"
                )


def test_primitive_options_groupbys(pd_es):
    options = {
        "cum_count": {"include_groupby_dataframes": ["log", "customers"]},
        "cum_sum": {"ignore_groupby_dataframes": ["sessions"]},
        "cum_mean": {
            "ignore_groupby_columns": {
                "customers": ["région_id"],
                "log": ["session_id"],
            },
        },
        "cum_min": {
            "include_groupby_columns": {"sessions": ["customer_id", "device_type"]},
        },
    }

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        max_depth=3,
        groupby_trans_primitives=["cum_sum", "cum_count", "cum_min", "cum_mean"],
        primitive_options=options,
    )
    features = dfs_obj.build_features()
    for f in features:
        if isinstance(f, GroupByTransformFeature):
            deps = f.groupby.get_dependencies(deep=True)
            df_names = [d.dataframe_name for d in deps] + [f.groupby.dataframe_name]
            columns = [d for d in deps if isinstance(d, IdentityFeature)]
            columns += [f.groupby] if isinstance(f.groupby, IdentityFeature) else []
        if isinstance(f.primitive, CumMean):
            for identity_groupby in columns:
                assert not (
                    identity_groupby.dataframe_name == "customers"
                    and identity_groupby.get_name() == "région_id"
                )
                assert not (
                    identity_groupby.dataframe_name == "log"
                    and identity_groupby.get_name() == "session_id"
                )
        if isinstance(f.primitive, CumCount):
            assert all([name in ["log", "customers"] for name in df_names])
        if isinstance(f.primitive, CumSum):
            assert "sessions" not in df_names
        if isinstance(f.primitive, CumMin):
            for identity_groupby in columns:
                if identity_groupby.dataframe_name == "sessions":
                    assert (
                        identity_groupby.get_name() == "customer_id"
                        or identity_groupby.get_name() == "device_type"
                    )


def test_primitive_options_multiple_inputs(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Dask and Spark EntitySets do not support various primitives used in this test",
        )
    too_many_options = {
        "mode": [{"include_dataframes": ["logs"]}, {"ignore_dataframes": ["sessions"]}],
    }
    error_msg = "Number of options does not match number of inputs for primitive mode"
    with pytest.raises(AssertionError, match=error_msg):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=["mode"],
            trans_primitives=[],
            primitive_options=too_many_options,
        )

    unknown_primitive = Trend()
    unknown_primitive.name = "unknown_primitive"
    unknown_primitive_option = {
        "unknown_primitive": [
            {"include_dataframes": ["logs"]},
            {"ignore_dataframes": ["sessions"]},
        ],
    }
    error_msg = "Unknown primitive with name 'unknown_primitive'"
    with pytest.raises(ValueError, match=error_msg):
        DeepFeatureSynthesis(
            target_dataframe_name="customers",
            entityset=es,
            agg_primitives=[unknown_primitive],
            trans_primitives=[],
            primitive_options=unknown_primitive_option,
        )

    options1 = {
        "trend": [
            {"include_dataframes": ["log"], "ignore_columns": {"log": ["value"]}},
            {"include_dataframes": ["log"], "include_columns": {"log": ["datetime"]}},
        ],
    }
    dfs_obj1 = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=["trend"],
        trans_primitives=[],
        primitive_options=options1,
    )
    features1 = dfs_obj1.build_features()
    for f in features1:
        deps = f.get_dependencies()
        df_names = [d.dataframe_name for d in deps]
        columns = [d.get_name() for d in deps]
        if f.primitive.name == "trend":
            assert all([df_name in ["log"] for df_name in df_names])
            assert "datetime" in columns
            if len(columns) == 2:
                assert "value" != columns[0]

    options2 = {
        Trend: [
            {"include_dataframes": ["log"], "ignore_columns": {"log": ["value"]}},
            {"include_dataframes": ["log"], "include_columns": {"log": ["datetime"]}},
        ],
    }
    dfs_obj2 = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=["trend"],
        trans_primitives=[],
        primitive_options=options2,
    )
    features2 = dfs_obj2.build_features()

    assert set(features2) == set(features1)


def test_primitive_options_class_names(es):
    options1 = {"mean": {"include_dataframes": ["customers"]}}

    options2 = {Mean: {"include_dataframes": ["customers"]}}

    bad_options = {
        "mean": {"include_dataframes": ["customers"]},
        Mean: {"ignore_dataframes": ["customers"]},
    }
    conflicting_error_text = "Multiple options found for primitive mean"

    primitives = [["mean"], [Mean]]
    options = [options1, options2]

    features = []
    for primitive in primitives:
        with pytest.raises(KeyError, match=conflicting_error_text):
            DeepFeatureSynthesis(
                target_dataframe_name="cohorts",
                entityset=es,
                agg_primitives=primitive,
                trans_primitives=[],
                primitive_options=bad_options,
            )
        for option in options:
            dfs_obj = DeepFeatureSynthesis(
                target_dataframe_name="cohorts",
                entityset=es,
                agg_primitives=primitive,
                trans_primitives=[],
                primitive_options=option,
            )
            features.append(set(dfs_obj.build_features()))

    for f in features[0]:
        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        if isinstance(f.primitive, Mean):
            assert all(df_name == "customers" for df_name in df_names)

    assert features[0] == features[1] == features[2] == features[3]


def test_primitive_options_instantiated_primitive(es):
    warning_msg = (
        "Options present for primitive instance and generic "
        "primitive class \\(mean\\), primitive instance will not use generic "
        "options"
    )

    skipna_mean = Mean(skipna=False)
    options = {
        skipna_mean: {"include_dataframes": ["stores"]},
        "mean": {"ignore_dataframes": ["stores"]},
    }
    with pytest.warns(UserWarning, match=warning_msg):
        dfs_obj = DeepFeatureSynthesis(
            target_dataframe_name="régions",
            entityset=es,
            agg_primitives=["mean", skipna_mean],
            trans_primitives=[],
            primitive_options=options,
        )

    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        df_names = [d.dataframe_name for d in deps]
        if f.primitive == skipna_mean:
            assert all(df_name == "stores" for df_name in df_names)
        elif isinstance(f.primitive, Mean):
            assert "stores" not in df_names


def test_primitive_options_commutative(es):
    class AddThree(TransformPrimitive):
        name = "add_three"
        input_types = [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(semantic_tags={"numeric"}),
        ]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        commutative = True
        compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

        def generate_name(self, base_feature_names):
            return "%s + %s + %s" % (
                base_feature_names[0],
                base_feature_names[1],
                base_feature_names[2],
            )

    options = {
        "add_numeric": [
            {"include_columns": {"log": ["value_2"]}},
            {"include_columns": {"log": ["value"]}},
        ],
        AddThree: [
            {"include_columns": {"log": ["value_2"]}},
            {"include_columns": {"log": ["value_many_nans"]}},
            {"include_columns": {"log": ["value"]}},
        ],
    }
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[AddNumeric, AddThree],
        primitive_options=options,
        max_depth=1,
    )
    features = dfs_obj.build_features()
    add_numeric = [f for f in features if isinstance(f.primitive, AddNumeric)]
    assert len(add_numeric) == 1
    deps = add_numeric[0].get_dependencies(deep=True)
    assert deps[0].get_name() == "value_2" and deps[1].get_name() == "value"

    add_three = [f for f in features if isinstance(f.primitive, AddThree)]
    assert len(add_three) == 1
    deps = add_three[0].get_dependencies(deep=True)
    assert (
        deps[0].get_name() == "value_2"
        and deps[1].get_name() == "value_many_nans"
        and deps[2].get_name() == "value"
    )


def test_primitive_options_include_over_exclude(es):
    options = {
        "mean": {"ignore_dataframes": ["stores"], "include_dataframes": ["stores"]},
    }
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="régions",
        entityset=es,
        agg_primitives=["mean"],
        trans_primitives=[],
        primitive_options=options,
    )

    features = dfs_obj.build_features()
    at_least_one_mean = False
    for f in features:
        deps = f.get_dependencies(deep=True)
        dataframes = [d.dataframe_name for d in deps]
        if isinstance(f.primitive, Mean):
            at_least_one_mean = True
            assert "stores" in dataframes
    assert at_least_one_mean


def test_primitive_ordering():
    # Test that the order of the input primitives impacts neither
    # which features are created nor their order
    es = make_ecommerce_entityset()

    trans_prims = [AddNumeric, Absolute, "divide_numeric", NotEqual, "is_null"]
    groupby_trans_prim = ["cum_mean", CumMin, CumSum]
    agg_prims = [NMostCommon(n=3), Sum, Mean, Mean(skipna=False), "min", "max"]
    where_prims = ["count", Sum]

    seed_num_chars = Feature(
        es["customers"].ww["favorite_quote"],
        primitive=NumCharacters,
    )
    seed_is_null = Feature(es["customers"].ww["age"], primitive=IsNull)
    seed_features = [seed_num_chars, seed_is_null]

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        trans_primitives=trans_prims,
        groupby_trans_primitives=groupby_trans_prim,
        agg_primitives=agg_prims,
        where_primitives=where_prims,
        seed_features=seed_features,
        max_features=-1,
        max_depth=2,
    )
    features1 = dfs_obj.build_features()

    trans_prims.reverse()
    groupby_trans_prim.reverse()
    agg_prims.reverse()
    where_prims.reverse()
    seed_features.reverse()

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="customers",
        entityset=es,
        trans_primitives=trans_prims,
        groupby_trans_primitives=groupby_trans_prim,
        agg_primitives=agg_prims,
        where_primitives=where_prims,
        seed_features=seed_features,
        max_features=-1,
        max_depth=2,
    )
    features2 = dfs_obj.build_features()

    assert len(features1) == len(features2)

    for i in range(len(features2)):
        assert features1[i].unique_name() == features2[i].unique_name()


def test_no_transform_stacking():
    df1 = pd.DataFrame({"id": [0, 1, 2, 3], "A": [0, 1, 2, 3]})
    df2 = pd.DataFrame(
        {"index": [0, 1, 2, 3], "first_id": [0, 1, 1, 3], "B": [99, 88, 77, 66]},
    )

    dataframes = {"first": (df1, "id"), "second": (df2, "index")}
    relationships = [("first", "id", "second", "first_id")]
    es = EntitySet("data", dataframes, relationships)

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="second",
        entityset=es,
        trans_primitives=["negate", "add_numeric"],
        agg_primitives=["sum"],
        max_depth=4,
    )
    feature_defs = dfs_obj.build_features()

    expected = [
        "first_id",
        "B",
        "-(B)",
        "first.A",
        "first.SUM(second.B)",
        "first.-(A)",
        "B + first.A",
        "first.SUM(second.-(B))",
        "first.A + SUM(second.B)",
        "first.-(SUM(second.B))",
        "B + first.SUM(second.B)",
        "first.A + SUM(second.-(B))",
        "first.SUM(second.-(B)) + SUM(second.B)",
        "first.-(SUM(second.-(B)))",
        "B + first.SUM(second.-(B))",
    ]

    assert len(feature_defs) == len(expected)

    for feature_name in expected:
        assert feature_with_name(feature_defs, feature_name)


def test_builds_seed_features_on_foreign_key_col(es):
    seed_feature_sessions = Feature(es["sessions"].ww["customer_id"], primitive=Negate)

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[],
        trans_primitives=[],
        max_depth=2,
        seed_features=[seed_feature_sessions],
    )

    features = dfs_obj.build_features()
    assert feature_with_name(features, "-(customer_id)")


def test_does_not_build_features_on_last_time_index_col(es):
    es.add_last_time_indexes()

    dfs_obj = DeepFeatureSynthesis(target_dataframe_name="log", entityset=es)

    features = dfs_obj.build_features()

    for feature in features:
        assert LTI_COLUMN_NAME not in feature.get_name()


def test_builds_features_using_all_input_types(es):
    if es.dataframe_type == Library.SPARK:
        pytest.skip("NumTrue primitive not compatible with Spark")
    new_log_df = es["log"]
    new_log_df.ww["purchased_nullable"] = es["log"]["purchased"]
    new_log_df.ww.set_types(logical_types={"purchased_nullable": "boolean_nullable"})
    es.replace_dataframe("log", new_log_df)

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        trans_primitives=[Not],
        max_depth=1,
    )
    trans_features = dfs_obj.build_features()
    assert feature_with_name(trans_features, "NOT(purchased)")
    assert feature_with_name(trans_features, "NOT(purchased_nullable)")

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=es,
        groupby_trans_primitives=[Not],
        max_depth=1,
    )
    groupby_trans_features = dfs_obj.build_features()
    assert feature_with_name(groupby_trans_features, "NOT(purchased) by session_id")
    assert feature_with_name(
        groupby_trans_features,
        "NOT(purchased_nullable) by session_id",
    )

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        trans_primitives=[],
        agg_primitives=[NumTrue],
    )
    agg_features = dfs_obj.build_features()
    assert feature_with_name(agg_features, "NUM_TRUE(log.purchased)")
    assert feature_with_name(agg_features, "NUM_TRUE(log.purchased_nullable)")


def test_make_groupby_features_with_depth_none(pd_es):
    # If max_depth is set to -1, it sets it to None internally, so this
    # test validates code paths that have a None max_depth
    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
        max_depth=-1,
    )
    features = dfs_obj.build_features()
    assert feature_with_name(features, "CUM_SUM(value) by session_id")


def test_check_stacking_when_building_transform_features(pd_es):
    class NewMean(Mean):
        name = "NEW_MEAN"
        base_of_exclude = [Absolute]

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[NewMean, "mean"],
        trans_primitives=["absolute"],
        max_depth=-1,
    )
    features = dfs_obj.build_features()
    assert number_of_features_with_name_like(features, "ABSOLUTE(MEAN") > 0
    assert number_of_features_with_name_like(features, "ABSOLUTE(NEW_MEAN") == 0


def test_check_stacking_when_building_groupby_features(pd_es):
    class NewMean(Mean):
        name = "NEW_MEAN"
        base_of_exclude = [CumSum]

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=[NewMean, "mean"],
        groupby_trans_primitives=["cum_sum"],
        max_depth=5,
    )
    features = dfs_obj.build_features()
    assert number_of_features_with_name_like(features, "CUM_SUM(MEAN") > 0
    assert number_of_features_with_name_like(features, "CUM_SUM(NEW_MEAN") == 0


def test_check_stacking_when_building_agg_features(pd_es):
    class NewAbsolute(Absolute):
        name = "NEW_ABSOLUTE"
        base_of_exclude = [Mean]

    dfs_obj = DeepFeatureSynthesis(
        target_dataframe_name="log",
        entityset=pd_es,
        agg_primitives=["mean"],
        trans_primitives=[NewAbsolute, "absolute"],
        max_depth=5,
    )
    features = dfs_obj.build_features()
    assert number_of_features_with_name_like(features, "MEAN(log.ABSOLUTE") > 0
    assert number_of_features_with_name_like(features, "MEAN(log.NEW_ABSOLUTE") == 0
