import copy
import sys

import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    IdentityFeature,
    TransformFeature
)
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
    NMostCommon,
    NotEqual,
    NumCharacters,
    NumUnique,
    Sum,
    TimeSincePrevious,
    TransformPrimitive,
    Trend,
    Year
)
from featuretools.synthesis import DeepFeatureSynthesis
from featuretools.tests.testing_utils import (
    feature_with_name,
    make_ecommerce_entityset
)
from featuretools.utils.gen_utils import Library, import_or_none, is_instance
from featuretools.variable_types import Datetime, Numeric

ks = import_or_none('databricks.koalas')


@pytest.fixture(params=['pd_transform_es', 'dask_transform_es', 'koalas_transform_es'])
def transform_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_transform_es():
    # Create dataframe
    df = pd.DataFrame({'a': [14, 12, 10], 'b': [False, False, True],
                       'b1': [True, True, False], 'b12': [4, 5, 6],
                       'P': [10, 15, 12]})
    es = ft.EntitySet(id='test')
    # Add dataframe to entityset
    es.entity_from_dataframe(entity_id='first', dataframe=df,
                             index='index',
                             make_index=True)

    return es


@pytest.fixture
def dask_transform_es(pd_transform_es):
    es = ft.EntitySet(id=pd_transform_es.id)
    for entity in pd_transform_es.entities:
        es.entity_from_dataframe(entity_id=entity.id,
                                 dataframe=dd.from_pandas(entity.df, npartitions=2),
                                 index=entity.index,
                                 variable_types=entity.variable_types)
    return es


@pytest.fixture
def koalas_transform_es(pd_transform_es):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    es = ft.EntitySet(id=pd_transform_es.id)
    for entity in pd_transform_es.entities:
        es.entity_from_dataframe(entity_id=entity.id,
                                 dataframe=ks.from_pandas(entity.df),
                                 index=entity.index,
                                 variable_types=entity.variable_types)
    return es


def test_makes_agg_features_from_str(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=['sum'],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'SUM(log.value)'))


def test_makes_agg_features_from_mixed_str(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Count, 'sum'],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'SUM(log.value)'))
    assert (feature_with_name(features, 'COUNT(log)'))


def test_case_insensitive(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=['MiN'],
                                   trans_primitives=['AbsOlute'])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'MIN(log.value)'))
    assert (feature_with_name(features, 'ABSOLUTE(MIN(log.value_many_nans))'))


def test_makes_agg_features(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'SUM(log.value)'))


def test_only_makes_supplied_agg_feat(es):
    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        max_depth=3,
    )
    dfs_obj = DeepFeatureSynthesis(agg_primitives=[Sum], **kwargs)

    features = dfs_obj.build_features()

    def find_other_agg_features(features):
        return [f for f in features
                if (isinstance(f, AggregationFeature) and
                    not isinstance(f.primitive, Sum)) or
                len([g for g in f.base_features
                     if isinstance(g, AggregationFeature) and
                     not isinstance(g.primitive, Sum)]) > 0]

    other_agg_features = find_other_agg_features(features)
    assert len(other_agg_features) == 0


def test_errors_unsupported_primitives(es):
    bad_trans_prim = CumSum()
    bad_agg_prim = NumUnique()
    bad_trans_prim.compatibility, bad_agg_prim.compatibility = [], []
    if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
        library = 'Dask'
    elif any(is_instance(entity.df, ks, 'DataFrame') for entity in es.entities):
        library = 'Koalas'
    else:
        library = 'pandas'
    error_text = "Selected primitives are incompatible with {} EntitySets: cum_sum, num_unique".format(library)
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(target_entity_id='sessions',
                             entityset=es,
                             agg_primitives=[bad_agg_prim],
                             trans_primitives=[bad_trans_prim])


def test_errors_unsupported_primitives_koalas(ks_es):
    bad_trans_prim = CumSum()
    bad_agg_prim = NumUnique()
    bad_trans_prim.koalas_compatible, bad_agg_prim.koalas_compatible = False, False
    error_text = "Selected primitives are incompatible with Koalas EntitySets: cum_sum"
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(target_entity_id='sessions',
                             entityset=ks_es,
                             agg_primitives=[bad_agg_prim],
                             trans_primitives=[bad_trans_prim])


def test_error_for_missing_target_entity(es):
    error_text = 'Provided target entity missing_entity does not exist in ecommerce'
    with pytest.raises(KeyError, match=error_text):
        DeepFeatureSynthesis(target_entity_id='missing_entity',
                             entityset=es,
                             agg_primitives=[Last],
                             trans_primitives=[],
                             ignore_entities=['log'])

    es_without_id = ft.EntitySet()
    error_text = 'Provided target entity missing_entity does not exist in entity set'
    with pytest.raises(KeyError, match=error_text):
        DeepFeatureSynthesis(target_entity_id='missing_entity',
                             entityset=es_without_id,
                             agg_primitives=[Last],
                             trans_primitives=[],
                             ignore_entities=['log'])


def test_ignores_entities(es):
    error_text = 'ignore_entities must be a list'
    with pytest.raises(TypeError, match=error_text):
        DeepFeatureSynthesis(target_entity_id='sessions',
                             entityset=es,
                             agg_primitives=[Sum],
                             trans_primitives=[],
                             ignore_entities='log')

    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[],
                                   ignore_entities=['log'])

    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        assert 'log' not in entities


def test_ignores_variables(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[],
                                   ignore_variables={'log': ['value']})
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        identities = [d for d in deps
                      if isinstance(d, IdentityFeature)]
        variables = [d.variable.id for d in identities
                     if d.entity.id == 'log']
        assert 'value' not in variables


def test_ignore_variables_input_type(es):
    error_msg = r'ignore_variables should be dict\[str -> list\]'  # need to use string literals to avoid regex params
    wrong_input_type = {'log': 'value'}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_entity_id='log',
            entityset=es,
            ignore_variables=wrong_input_type,
        )


def test_ignore_variables_with_nonstring_values(es):
    error_msg = 'list values should be of type str'
    wrong_input_list = {'log': ['a', 'b', 3]}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_entity_id='log',
            entityset=es,
            ignore_variables=wrong_input_list,
        )


def test_ignore_variables_with_nonstring_keys(es):
    error_msg = r'ignore_variables should be dict\[str -> list\]'  # need to use string literals to avoid regex params
    wrong_input_keys = {1: ['a', 'b', 'c']}
    with pytest.raises(TypeError, match=error_msg):
        DeepFeatureSynthesis(
            target_entity_id='log',
            entityset=es,
            ignore_variables=wrong_input_keys,
        )


def test_makes_dfeatures(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'customers.age'))


def test_makes_trans_feat(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=[Hour])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'HOUR(datetime)'))


def test_handles_diff_entity_groupby(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   groupby_trans_primitives=[Diff])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'DIFF(value) by session_id'))
    assert (feature_with_name(features, 'DIFF(value) by product_id'))


def test_handles_time_since_previous_entity_groupby(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   groupby_trans_primitives=[TimeSincePrevious])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'TIME_SINCE_PREVIOUS(datetime) by session_id'))

# M TODO
# def test_handles_cumsum_entity_groupby(pd_es):
#     dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
#                                    entityset=pd_es,
#                                    agg_primitives=[],
#                                    trans_primitives=[CumMean])

#     features = dfs_obj.build_features()
#     assert (feature_with_name(features, u'customers.CUM_MEAN(age by région_id)'))


def test_only_makes_supplied_trans_feat(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=[Hour])

    features = dfs_obj.build_features()
    other_trans_features = [f for f in features
                            if (isinstance(f, TransformFeature) and
                                not isinstance(f.primitive, Hour)) or
                            len([g for g in f.base_features
                                 if isinstance(g, TransformFeature) and
                                 not isinstance(g.primitive, Hour)]) > 0]
    assert len(other_trans_features) == 0


def test_makes_dfeatures_of_agg_primitives(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=['max'],
                                   trans_primitives=[])
    features = dfs_obj.build_features()

    assert (feature_with_name(features,
                              'customers.MAX(log.value)'))


def test_makes_agg_features_of_trans_primitives(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Mean],
                                   trans_primitives=[NumCharacters])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, 'MEAN(log.NUM_CHARACTERS(comments))'))


def test_makes_agg_features_with_where(es):
    # TODO: Update to work with Dask and Koalas `es` fixture when issue #978 is closed
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support add_interesting_values")
    es.add_interesting_values()

    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Count],
                                   where_primitives=[Count],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert (feature_with_name(features,
                              'COUNT(log WHERE priority_level = 0)'))

    # make sure they are made using direct features too
    assert (feature_with_name(features,
                              'COUNT(log WHERE products.department = food)'))


def test_make_groupby_features(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   trans_primitives=[],
                                   groupby_trans_primitives=['cum_sum'])
    features = dfs_obj.build_features()
    assert (feature_with_name(features,
                              "CUM_SUM(value) by session_id"))


def test_make_indirect_groupby_features(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   trans_primitives=[],
                                   groupby_trans_primitives=['cum_sum'])
    features = dfs_obj.build_features()
    assert (feature_with_name(features,
                              "CUM_SUM(products.rating) by session_id"))


def test_make_groupby_features_with_id(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   trans_primitives=[],
                                   groupby_trans_primitives=['cum_count'])
    features = dfs_obj.build_features()
    assert (feature_with_name(features, "CUM_COUNT(customer_id) by customer_id"))


def test_make_groupby_features_with_diff_id(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='customers',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   trans_primitives=[],
                                   groupby_trans_primitives=['cum_count'])
    features = dfs_obj.build_features()
    groupby_with_diff_id = u"CUM_COUNT(cohort) by région_id"
    assert (feature_with_name(features, groupby_with_diff_id))


def test_make_groupby_features_with_agg(pd_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='cohorts',
                                   entityset=pd_es,
                                   agg_primitives=['sum'],
                                   trans_primitives=[],
                                   groupby_trans_primitives=['cum_sum'])
    features = dfs_obj.build_features()
    agg_on_groupby_name = u"SUM(customers.CUM_SUM(age) by région_id)"
    assert (feature_with_name(features, agg_on_groupby_name))


def test_bad_groupby_feature(es):
    msg = "Unknown transform primitive max"
    with pytest.raises(ValueError, match=msg):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['sum'],
                             trans_primitives=[],
                             groupby_trans_primitives=['max'])


def test_abides_by_max_depth_param(es):
    for i in [1, 2, 3]:
        dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                       entityset=es,
                                       agg_primitives=[Sum],
                                       trans_primitives=[],
                                       max_depth=i)

        features = dfs_obj.build_features()
        for f in features:
            # last feature is identity feature which doesn't count
            assert (f.get_depth() <= i + 1)


def test_drop_contains(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[],
                                   max_depth=1,
                                   seed_features=[],
                                   drop_contains=[])
    features = dfs_obj.build_features()
    to_drop = features[0]
    partial_name = to_drop.get_name()[:5]
    dfs_drop = DeepFeatureSynthesis(target_entity_id='sessions',
                                    entityset=es,
                                    agg_primitives=[Sum],
                                    trans_primitives=[],
                                    max_depth=1,
                                    seed_features=[],
                                    drop_contains=[partial_name])
    features = dfs_drop.build_features()
    assert to_drop.get_name() not in [f.get_name() for f in features]


def test_drop_exact(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[],
                                   max_depth=1,
                                   seed_features=[],
                                   drop_exact=[])
    features = dfs_obj.build_features()
    to_drop = features[0]
    name = to_drop.get_name()
    dfs_drop = DeepFeatureSynthesis(target_entity_id='sessions',
                                    entityset=es,
                                    agg_primitives=[Sum],
                                    trans_primitives=[],
                                    max_depth=1,
                                    seed_features=[],
                                    drop_exact=[name])
    features = dfs_drop.build_features()
    assert name not in [f.get_name() for f in features]


def test_seed_features(es):
    seed_feature_sessions = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count) > 2
    seed_feature_log = ft.Feature(es['log']['comments'], primitive=NumCharacters)
    session_agg = ft.Feature(seed_feature_log, parent_entity=es['sessions'], primitive=Mean)
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Mean],
                                   trans_primitives=[],
                                   max_depth=2,
                                   seed_features=[seed_feature_sessions,
                                                  seed_feature_log])
    features = dfs_obj.build_features()
    assert seed_feature_sessions.get_name() in [f.get_name()
                                                for f in features]
    assert session_agg.get_name() in [f.get_name() for f in features]


def test_does_not_make_agg_of_direct_of_target_entity(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the Last primitive")

    count_sessions = ft.Feature(es['sessions']['id'], parent_entity=es['customers'], primitive=Count)
    dfs_obj = DeepFeatureSynthesis(target_entity_id='customers',
                                   entityset=es,
                                   agg_primitives=[Last],
                                   trans_primitives=[],
                                   max_depth=2,
                                   seed_features=[count_sessions])
    features = dfs_obj.build_features()
    # this feature is meaningless because customers.COUNT(sessions) is already defined on
    # the customers entity
    assert not feature_with_name(features, 'LAST(sessions.customers.COUNT(sessions))')
    assert not feature_with_name(features, 'LAST(sessions.customers.age)')


def test_dfs_builds_on_seed_features_more_than_max_depth(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the Last and Mode primitives")

    seed_feature_sessions = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    seed_feature_log = ft.Feature(es['log']['datetime'], primitive=Hour)
    session_agg = ft.Feature(seed_feature_log, parent_entity=es['sessions'], primitive=Last)

    # Depth of this feat is 2 relative to session_agg, the seed feature,
    # which is greater than max_depth so it shouldn't be built
    session_agg_trans = DirectFeature(ft.Feature(session_agg, parent_entity=es['customers'], primitive=Mode),
                                      es['sessions'])
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Last, Count],
                                   trans_primitives=[],
                                   max_depth=1,
                                   seed_features=[seed_feature_sessions,
                                                  seed_feature_log])
    features = dfs_obj.build_features()
    assert seed_feature_sessions.get_name() in [f.get_name()
                                                for f in features]
    assert session_agg.get_name() in [f.get_name() for f in features]
    assert session_agg_trans.get_name() not in [f.get_name()
                                                for f in features]


def test_allowed_paths(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the Last primitive")

    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        agg_primitives=[Last],
        trans_primitives=[],
        max_depth=2,
        seed_features=[]
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features_unconstrained = dfs_unconstrained.build_features()

    unconstrained_names = [f.get_name() for f in features_unconstrained]
    customers_session_feat = ft.Feature(es['sessions']['device_type'], parent_entity=es['customers'], primitive=Last)
    customers_session_log_feat = ft.Feature(es['log']['value'], parent_entity=es['customers'], primitive=Last)
    assert customers_session_feat.get_name() in unconstrained_names
    assert customers_session_log_feat.get_name() in unconstrained_names

    dfs_constrained = DeepFeatureSynthesis(allowed_paths=[['customers',
                                                           'sessions']],
                                           **kwargs)
    features = dfs_constrained.build_features()
    names = [f.get_name() for f in features]
    assert customers_session_feat.get_name() in names
    assert customers_session_log_feat.get_name() not in names


def test_max_features(es):
    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        agg_primitives=[Sum],
        trans_primitives=[],
        max_depth=2,
        seed_features=[]
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features_unconstrained = dfs_unconstrained.build_features()
    dfs_unconstrained_with_arg = DeepFeatureSynthesis(max_features=-1,
                                                      **kwargs)
    feats_unconstrained_with_arg = dfs_unconstrained_with_arg.build_features()
    dfs_constrained = DeepFeatureSynthesis(max_features=1, **kwargs)
    features = dfs_constrained.build_features()
    assert len(features_unconstrained) == len(feats_unconstrained_with_arg)
    assert len(features) == 1


def test_where_primitives(es):
    es['sessions']['device_type'].interesting_values = [0]
    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        agg_primitives=[Count, Sum],
        trans_primitives=[Absolute],
        max_depth=3,
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    dfs_constrained = DeepFeatureSynthesis(where_primitives=['sum'], **kwargs)
    features_unconstrained = dfs_unconstrained.build_features()
    features = dfs_constrained.build_features()

    where_feats_unconstrained = [f for f in features_unconstrained
                                 if isinstance(f, AggregationFeature) and f.where is not None]
    where_feats = [f for f in features
                   if isinstance(f, AggregationFeature) and f.where is not None]

    assert len(where_feats_unconstrained) >= 1

    assert len([f for f in where_feats_unconstrained
                if isinstance(f.primitive, Sum)]) == 0
    assert len([f for f in where_feats_unconstrained
                if isinstance(f.primitive, Count)]) > 0

    assert len([f for f in where_feats
                if isinstance(f.primitive, Sum)]) > 0
    assert len([f for f in where_feats
                if isinstance(f.primitive, Count)]) == 0
    assert len([d for f in where_feats
                for d in f.get_dependencies(deep=True)
                if isinstance(d.primitive, Absolute)]) > 0


def test_stacking_where_primitives(es):
    # TODO: Update to work with Dask supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask and Koalas EntitySets do not support the Last primitive")
    es = copy.deepcopy(es)
    es['sessions']['device_type'].interesting_values = [0]
    es['log']['product_id'].interesting_values = ["coke_zero"]
    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        agg_primitives=[Count, Last],
        max_depth=3,
    )
    dfs_where_stack_limit_1 = DeepFeatureSynthesis(where_primitives=['last', Count],
                                                   **kwargs)
    dfs_where_stack_limit_2 = DeepFeatureSynthesis(where_primitives=['last', Count],
                                                   where_stacking_limit=2,
                                                   **kwargs)
    stack_limit_1_features = dfs_where_stack_limit_1.build_features()
    stack_limit_2_features = dfs_where_stack_limit_2.build_features()

    where_stack_1_feats = [f for f in stack_limit_1_features
                           if isinstance(f, AggregationFeature) and f.where is not None]
    where_stack_2_feats = [f for f in stack_limit_2_features
                           if isinstance(f, AggregationFeature) and f.where is not None]

    assert len(where_stack_1_feats) >= 1
    assert len(where_stack_2_feats) >= 1

    assert len([f for f in where_stack_1_feats
                if isinstance(f.primitive, Last)]) > 0
    assert len([f for f in where_stack_1_feats
                if isinstance(f.primitive, Count)]) > 0

    assert len([f for f in where_stack_2_feats
                if isinstance(f.primitive, Last)]) > 0
    assert len([f for f in where_stack_2_feats
                if isinstance(f.primitive, Count)]) > 0

    stacked_where_limit_1_feats = []
    stacked_where_limit_2_feats = []
    where_double_where_tuples = [
        (where_stack_1_feats, stacked_where_limit_1_feats),
        (where_stack_2_feats, stacked_where_limit_2_feats)
    ]
    for where_list, double_where_list in where_double_where_tuples:
        for feature in where_list:
            for base_feat in feature.base_features:
                if isinstance(base_feat, AggregationFeature) and base_feat.where is not None:
                    double_where_list.append(feature)

    assert len(stacked_where_limit_1_feats) == 0
    assert len(stacked_where_limit_2_feats) > 0


def test_where_different_base_feats(es):
    es['sessions']['device_type'].interesting_values = [0]

    kwargs = dict(
        target_entity_id='customers',
        entityset=es,
        agg_primitives=[Sum, Count],
        where_primitives=[Sum, Count],
        max_depth=3,
    )
    dfs_unconstrained = DeepFeatureSynthesis(**kwargs)
    features = dfs_unconstrained.build_features()
    where_feats = [f.unique_name() for f in features
                   if isinstance(f, AggregationFeature) and f.where is not None]
    not_where_feats = [f.unique_name() for f in features
                       if isinstance(f, AggregationFeature) and f.where is None]
    for name in not_where_feats:
        assert name not in where_feats


def test_dfeats_where(es):
    # TODO: Update to work with Dask `es` fixture when issue #978 is closed
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask and Koalas EntitySets do not support add_interesting_values")
    es.add_interesting_values()

    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[Count],
                                   trans_primitives=[])

    features = dfs_obj.build_features()

    # test to make sure we build direct features of agg features with where clause
    assert (feature_with_name(
        features, 'customers.COUNT(log WHERE priority_level = 0)'))

    assert (feature_with_name(
        features, 'COUNT(log WHERE products.department = electronics)'))


def test_commutative(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=[Sum],
                                   trans_primitives=[AddNumeric],
                                   max_depth=3)
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
    feature_defs = ft.dfs(entityset=transform_es, target_entity='first',
                          trans_primitives=['and', 'add_numeric', 'or'],
                          features_only=True)

    # Check for correct ordering of features
    assert feature_with_name(feature_defs, 'a')
    assert feature_with_name(feature_defs, 'b')
    assert feature_with_name(feature_defs, 'b1')
    assert feature_with_name(feature_defs, 'b12')
    assert feature_with_name(feature_defs, 'P')
    assert feature_with_name(feature_defs, 'AND(b, b1)')
    assert not feature_with_name(feature_defs, 'AND(b1, b)')  # make sure it doesn't exist the other way
    assert feature_with_name(feature_defs, 'a + P')
    assert feature_with_name(feature_defs, 'b12 + P')
    assert feature_with_name(feature_defs, 'a + b12')
    assert feature_with_name(feature_defs, 'OR(b, b1)')


def test_transform_no_stack_agg(es):
    # TODO: Update to work with Dask and Koalas supported primitives
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the NMostCommon primitive")
    feature_defs = ft.dfs(entityset=es,
                          target_entity="customers",
                          agg_primitives=[NMostCommon],
                          trans_primitives=[NotEqual],
                          max_depth=3,
                          features_only=True)
    assert not feature_with_name(feature_defs, 'id != N_MOST_COMMON(sessions.device_type)')


def test_intialized_trans_prim(es):
    prim = IsIn(list_of_outputs=['coke zero'])
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=[prim])

    features = dfs_obj.build_features()
    assert (feature_with_name(features, "product_id.isin(['coke zero'])"))


def test_initialized_agg_prim(es):
    # TODO: Update to work with Dask and Koalas supported primitives
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the NMostCommon primitive")
    ThreeMost = NMostCommon(n=3)
    dfs_obj = DeepFeatureSynthesis(target_entity_id="sessions",
                                   entityset=es,
                                   agg_primitives=[ThreeMost],
                                   trans_primitives=[])
    features = dfs_obj.build_features()
    assert (feature_with_name(features, "N_MOST_COMMON(log.product_id)"))


def test_return_variable_types(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask and Koalas EntitySets do not support the NMostCommon primitive")
    dfs_obj = DeepFeatureSynthesis(target_entity_id="sessions",
                                   entityset=es,
                                   agg_primitives=[Count, NMostCommon],
                                   trans_primitives=[Absolute, Hour, IsIn])

    discrete = ft.variable_types.Discrete
    numeric = ft.variable_types.Numeric
    datetime = ft.variable_types.Datetime

    f1 = dfs_obj.build_features(return_variable_types=None)
    f2 = dfs_obj.build_features(return_variable_types=[discrete])
    f3 = dfs_obj.build_features(return_variable_types="all")
    f4 = dfs_obj.build_features(return_variable_types=[datetime])

    f1_types = set([f.variable_type for f in f1])
    f2_types = set([f.variable_type for f in f2])
    f3_types = set([f.variable_type for f in f3])
    f4_types = set([f.variable_type for f in f4])

    assert(discrete in f1_types)
    assert(numeric in f1_types)
    assert(datetime not in f2_types)

    assert(discrete in f2_types)
    assert(numeric not in f2_types)
    assert(datetime not in f2_types)

    assert(discrete in f3_types)
    assert(numeric in f3_types)
    assert(datetime in f3_types)

    assert(discrete not in f4_types)
    assert(numeric not in f4_types)
    assert(datetime in f4_types)


def test_checks_primitives_correct_type(es):
    error_text = "Primitive <class \\'featuretools\\.primitives\\.standard\\."\
                 "transform_primitive\\.Hour\\'> in agg_primitives is not an "\
                 "aggregation primitive"
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(target_entity_id="sessions",
                             entityset=es,
                             agg_primitives=[Hour],
                             trans_primitives=[])

    error_text = "Primitive <class \\'featuretools\\.primitives\\.standard\\."\
                 "aggregation_primitives\\.Sum\\'> in trans_primitives or "\
                 "groupby_trans_primitives is not a transform primitive"
    with pytest.raises(ValueError, match=error_text):
        DeepFeatureSynthesis(target_entity_id="sessions",
                             entityset=es,
                             agg_primitives=[],
                             trans_primitives=[Sum])


def test_makes_agg_features_along_multiple_paths(diamond_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='regions',
                                   entityset=diamond_es,
                                   agg_primitives=['mean'],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert feature_with_name(features, 'MEAN(customers.transactions.amount)')
    assert feature_with_name(features, 'MEAN(stores.transactions.amount)')


def test_makes_direct_features_through_multiple_relationships(games_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='games',
                                   entityset=games_es,
                                   agg_primitives=['mean'],
                                   trans_primitives=[])

    features = dfs_obj.build_features()

    teams = ['home', 'away']
    for forward in teams:
        for backward in teams:
            for var in teams:
                f = 'teams[%s_team_id].MEAN(games[%s_team_id].%s_team_score)' \
                    % (forward, backward, var)
                assert feature_with_name(features, f)


def test_stacks_multioutput_features(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the NumUnique and NMostCommon primitives")

    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [Datetime]
        return_type = Numeric
        number_output_features = 6

        def get_function(self):
            def test_f(x):
                times = pd.Series(x)
                units = ["year", "month", "day", "hour", "minute", "second"]
                return [times.apply(lambda x: getattr(x, unit)) for unit in units]
            return test_f

    feat = ft.dfs(entityset=es,
                  target_entity="customers",
                  agg_primitives=[NumUnique, NMostCommon(n=3)],
                  trans_primitives=[TestTime, Diff],
                  max_depth=4,
                  features_only=True
                  )

    for i in range(3):
        f = 'NUM_UNIQUE(sessions.N_MOST_COMMON(log.countrycode)[%d])' % i
        assert feature_with_name(feat, f)


def test_seed_multi_output_feature_stacking(es):
    # TODO: Update to work with Dask and Koalas supported primitive
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask EntitySets do not support the NMostCommon and NumUnique primitives")
    threecommon = NMostCommon(3)
    tc = ft.Feature(es['log']['product_id'], parent_entity=es["sessions"], primitive=threecommon)

    fm, feat = ft.dfs(entityset=es,
                      target_entity="customers",
                      seed_features=[tc],
                      agg_primitives=[NumUnique],
                      trans_primitives=[],
                      max_depth=4
                      )

    for i in range(3):
        f = 'NUM_UNIQUE(sessions.N_MOST_COMMON(log.product_id)[%d])' % i
        assert feature_with_name(feat, f)


def test_makes_direct_features_along_multiple_paths(diamond_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='transactions',
                                   entityset=diamond_es,
                                   max_depth=3,
                                   agg_primitives=[],
                                   trans_primitives=[])

    features = dfs_obj.build_features()
    assert feature_with_name(features, 'customers.regions.name')
    assert feature_with_name(features, 'stores.regions.name')


def test_does_not_make_trans_of_single_direct_feature(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='sessions',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=['weekday'],
                                   max_depth=2)

    features = dfs_obj.build_features()

    assert not feature_with_name(features, 'WEEKDAY(customers.signup_date)')
    assert feature_with_name(features, 'customers.WEEKDAY(signup_date)')


def test_makes_trans_of_multiple_direct_features(diamond_es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='transactions',
                                   entityset=diamond_es,
                                   agg_primitives=['mean'],
                                   trans_primitives=[Equal],
                                   max_depth=4)

    features = dfs_obj.build_features()

    # Make trans of direct and non-direct
    assert feature_with_name(features, 'amount = stores.MEAN(transactions.amount)')

    # Make trans of direct features on different entities
    assert feature_with_name(features, 'customers.MEAN(transactions.amount) = stores.square_ft')

    # Make trans of direct features on same entity with different paths.
    assert feature_with_name(features, 'customers.regions.name = stores.regions.name')

    # Don't make trans of direct features with same path.
    assert not feature_with_name(features, 'stores.square_ft = stores.MEAN(transactions.amount)')
    assert not feature_with_name(features, 'stores.MEAN(transactions.amount) = stores.square_ft')

    # The naming of the below is confusing but this is a direct feature of a transform.
    assert feature_with_name(features, 'stores.MEAN(transactions.amount) = square_ft')


def test_makes_direct_of_agg_of_trans_on_target(es):
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=['mean'],
                                   trans_primitives=[Absolute],
                                   max_depth=3)

    features = dfs_obj.build_features()
    assert feature_with_name(features, 'sessions.MEAN(log.ABSOLUTE(value))')


def test_primitive_options_errors(es):
    wrong_key_options = {'mean': {'ignore_entity': ['sessions']}}
    wrong_type_list = {'mean': {'ignore_entities': 'sessions'}}
    wrong_type_dict = {'mean':
                       {'ignore_variables': {'sessions': 'product_id'}}}
    conflicting_primitive_options = {('count', 'mean'):
                                     {'ignore_entities': ['sessions']},
                                     'mean': {'include_entities': ['sessions']}}
    invalid_entity = {'mean': {'include_entities': ['invalid_entity']}}
    invalid_variable_entity = {'mean': {'include_variables': {'invalid_entity': ['product_id']}}}
    invalid_variable = {'mean': {'include_variables': {'sessions': ['invalid_variable']}}}
    key_error_text = "Unrecognized primitive option 'ignore_entity' for mean"
    list_error_text = "Incorrect type formatting for 'ignore_entities' for mean"
    dict_error_text = "Incorrect type formatting for 'ignore_variables' for mean"
    conflicting_error_text = "Multiple options found for primitive mean"
    invalid_entity_warning = "Entity 'invalid_entity' not in entityset"
    invalid_variable_warning = "Variable 'invalid_variable' not in entity 'sessions'"
    with pytest.raises(KeyError, match=key_error_text):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=wrong_key_options)
    with pytest.raises(TypeError, match=list_error_text):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=wrong_type_list)
    with pytest.raises(TypeError, match=dict_error_text):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=wrong_type_dict)
    with pytest.raises(KeyError, match=conflicting_error_text):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=conflicting_primitive_options)
    with pytest.warns(UserWarning, match=invalid_entity_warning) as record:
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=invalid_entity)
    assert len(record) == 1
    with pytest.warns(UserWarning, match=invalid_entity_warning) as record:
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=invalid_variable_entity)
    assert len(record) == 1
    with pytest.warns(UserWarning, match=invalid_variable_warning) as record:
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mean'],
                             trans_primitives=[],
                             primitive_options=invalid_variable)
    assert len(record) == 1


def test_primitive_options(es):
    options = {'sum': {'include_variables': {'customers': ['age']}},
               'mean': {'include_entities': ['customers']},
               'mode': {'ignore_entities': ['sessions']},
               'num_unique': {'ignore_variables': {'customers': ['engagement_level']}}}
    dfs_obj = DeepFeatureSynthesis(target_entity_id='cohorts',
                                   entityset=es,
                                   primitive_options=options)
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        variables = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Sum):
            for identity_base in variables:
                if identity_base.entity.id == 'customers':
                    assert identity_base.get_name() == 'age'
        if isinstance(f.primitive, Mean):
            assert all([entity in ['customers'] for entity in entities])
        if isinstance(f.primitive, Mode):
            assert 'sessions' not in entities
        if isinstance(f.primitive, NumUnique):
            for identity_base in variables:
                assert not (identity_base.entity.id == 'customers' and
                            identity_base.get_name() == 'engagement_level')

    options = {'month': {'ignore_variables': {'customers': ['date_of_birth']}},
               'day': {'include_variables': {'customers': ['signup_date', 'upgrade_date']}},
               'num_characters': {'ignore_entities': ['customers']},
               'year': {'include_entities': ['customers']}}
    dfs_obj = DeepFeatureSynthesis(target_entity_id='customers',
                                   entityset=es,
                                   agg_primitives=[],
                                   ignore_entities=['cohort'],
                                   primitive_options=options)
    features = dfs_obj.build_features()
    assert not any([isinstance(f, NumCharacters) for f in features])
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        variables = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Month):
            for identity_base in variables:
                assert not (identity_base.entity.id == 'customers' and
                            identity_base.get_name() == 'date_of_birth')
        if isinstance(f.primitive, Day):
            for identity_base in variables:
                if identity_base.entity.id == 'customers':
                    assert identity_base.get_name() == 'signup_date' or \
                        identity_base.get_name() == 'upgrade_date'
        if isinstance(f.primitive, Year):
            assert all([entity in ['customers'] for entity in entities])


def test_primitive_options_with_globals(es):
    # non-overlapping ignore_entities
    options = {'mode': {'ignore_entities': ['sessions']}}
    dfs_obj = DeepFeatureSynthesis(target_entity_id='cohorts',
                                   entityset=es,
                                   ignore_entities=[u'régions'],
                                   primitive_options=options)
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        assert u'régions' not in entities
        if isinstance(f.primitive, Mode):
            assert 'sessions' not in entities

    # non-overlapping ignore_variables
    options = {'num_unique': {'ignore_variables': {'customers': ['engagement_level']}}}
    dfs_obj = DeepFeatureSynthesis(target_entity_id='customers',
                                   entityset=es,
                                   ignore_variables={'customers': [u'région_id']},
                                   primitive_options=options)
    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        variables = [d for d in deps if isinstance(d, IdentityFeature)]
        for identity_base in variables:
            assert not (identity_base.entity.id == 'customers' and
                        identity_base.get_name() == u'région_id')
        if isinstance(f.primitive, NumUnique):
            for identity_base in variables:
                assert not (identity_base.entity.id == 'customers' and
                            identity_base.get_name() == 'engagement_level')

    # Overlapping globals/options with ignore_entities
    options = {'mode': {'include_entities': ['sessions', 'customers'],
                        'ignore_variables': {'customers': [u'région_id']}},
               'num_unique': {'include_entities': ['sessions', 'customers'],
                              'include_variables': {'sessions': ['device_type'],
                                                    'customers': ['age']}},
               'month': {'ignore_variables': {'cohorts': ['cohort_end']}}}
    dfs_obj = DeepFeatureSynthesis(target_entity_id='cohorts',
                                   entityset=es,
                                   ignore_entities=['sessions'],
                                   ignore_variables={'customers': ['age']},
                                   primitive_options=options)
    features = dfs_obj.build_features()
    for f in features:
        assert f.primitive.name != 'month'
        # ignoring cohorts means no features are created
        assert not isinstance(f.primitive, Month)

        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        variables = [d for d in deps if isinstance(d, IdentityFeature)]
        if isinstance(f.primitive, Mode):
            assert [all([entity in ['sessions', 'customers'] for entity in entities])]
            for identity_base in variables:
                assert not (identity_base.entity.id == 'customers' and
                            (identity_base.get_name() == 'age' or
                             identity_base.get_name() == u'région_id'))
        elif isinstance(f.primitive, NumUnique):
            assert [all([entity in ['sessions', 'customers'] for entity in entities])]
            for identity_base in variables:
                if identity_base.entity.id == 'sessions':
                    assert identity_base.get_name() == 'device_type'
        # All other primitives ignore 'sessions' and 'age'
        else:
            assert 'sessions' not in entities
            for identity_base in variables:
                assert not (identity_base.entity.id == 'customers' and
                            identity_base.get_name() == 'age')


def test_primitive_options_groupbys(pd_es):
    options = {'cum_count': {'include_groupby_entities': ['log', 'customers']},
               'cum_sum': {'ignore_groupby_entities': ['sessions']},
               'cum_mean': {'ignore_groupby_variables': {'customers': [u'région_id'],
                                                         'log': ['session_id']}},
               'cum_min': {'include_groupby_variables': {'sessions': ['customer_id', 'device_type']}}}

    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=pd_es,
                                   agg_primitives=[],
                                   trans_primitives=[],
                                   max_depth=3,
                                   groupby_trans_primitives=['cum_sum',
                                                             'cum_count',
                                                             'cum_min',
                                                             'cum_mean'],
                                   primitive_options=options)
    features = dfs_obj.build_features()
    for f in features:
        if isinstance(f, ft.GroupByTransformFeature):
            deps = f.groupby.get_dependencies(deep=True)
            entities = [d.entity.id for d in deps] + [f.groupby.entity.id]
            variables = [d for d in deps if isinstance(d, IdentityFeature)]
            variables += [f.groupby] if isinstance(f.groupby, IdentityFeature) else []
        if isinstance(f.primitive, CumMean):
            for identity_groupby in variables:
                assert not (identity_groupby.entity.id == 'customers' and
                            identity_groupby.get_name() == u'région_id')
                assert not (identity_groupby.entity.id == 'log' and
                            identity_groupby.get_name() == 'session_id')
        if isinstance(f.primitive, CumCount):
            assert all([entity in ['log', 'customers'] for entity in entities])
        if isinstance(f.primitive, CumSum):
            assert 'sessions' not in entities
        if isinstance(f.primitive, CumMin):
            for identity_groupby in variables:
                if identity_groupby.entity.id == 'sessions':
                    assert identity_groupby.get_name() == 'customer_id' or\
                        identity_groupby.get_name() == 'device_type'


def test_primitive_options_multiple_inputs(es):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Dask and Koalas EntitySets do not support various primitives used in this test")
    too_many_options = {'mode': [{'include_entities': ['logs']},
                                 {'ignore_entities': ['sessions']}]}
    error_msg = "Number of options does not match number of inputs for primitive mode"
    with pytest.raises(AssertionError, match=error_msg):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=['mode'],
                             trans_primitives=[],
                             primitive_options=too_many_options)

    unknown_primitive = Trend()
    unknown_primitive.name = 'unknown_primitive'
    unknown_primitive_option = {'unknown_primitive': [{'include_entities': ['logs']},
                                                      {'ignore_entities': ['sessions']}]}
    error_msg = "Unknown primitive with name 'unknown_primitive'"
    with pytest.raises(ValueError, match=error_msg):
        DeepFeatureSynthesis(target_entity_id='customers',
                             entityset=es,
                             agg_primitives=[unknown_primitive],
                             trans_primitives=[],
                             primitive_options=unknown_primitive_option)

    options1 = {'trend': [{'include_entities': ['log'],
                           'ignore_variables': {'log': ['value']}},
                          {'include_entities': ['log'],
                           'include_variables': {'log': ['datetime']}}]}
    dfs_obj1 = DeepFeatureSynthesis(target_entity_id='sessions',
                                    entityset=es,
                                    agg_primitives=['trend'],
                                    trans_primitives=[],
                                    primitive_options=options1)
    features1 = dfs_obj1.build_features()
    for f in features1:
        deps = f.get_dependencies()
        entities = [d.entity.id for d in deps]
        variables = [d.get_name() for d in deps]
        if f.primitive.name == 'trend':
            assert all([entity in ['log'] for entity in entities])
            assert 'datetime' in variables
            if len(variables) == 2:
                assert 'value' != variables[0]

    options2 = {Trend: [{'include_entities': ['log'],
                         'ignore_variables': {'log': ['value']}},
                        {'include_entities': ['log'],
                         'include_variables': {'log': ['datetime']}}]}
    dfs_obj2 = DeepFeatureSynthesis(target_entity_id='sessions',
                                    entityset=es,
                                    agg_primitives=['trend'],
                                    trans_primitives=[],
                                    primitive_options=options2)
    features2 = dfs_obj2.build_features()

    assert set(features2) == set(features1)


def test_primitive_options_class_names(es):
    options1 = {
        'mean': {'include_entities': ['customers']}
    }

    options2 = {
        Mean: {'include_entities': ['customers']}
    }

    bad_options = {
        'mean': {'include_entities': ['customers']},
        Mean: {'ignore_entities': ['customers']}
    }
    conflicting_error_text = "Multiple options found for primitive mean"

    primitives = [['mean'], [Mean]]
    options = [options1, options2]

    features = []
    for primitive in primitives:
        with pytest.raises(KeyError, match=conflicting_error_text):
            DeepFeatureSynthesis(target_entity_id='cohorts',
                                 entityset=es,
                                 agg_primitives=primitive,
                                 trans_primitives=[],
                                 primitive_options=bad_options)
        for option in options:
            dfs_obj = DeepFeatureSynthesis(target_entity_id='cohorts',
                                           entityset=es,
                                           agg_primitives=primitive,
                                           trans_primitives=[],
                                           primitive_options=option)
            features.append(set(dfs_obj.build_features()))

    for f in features[0]:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        if isinstance(f.primitive, Mean):
            assert all(entity == 'customers' for entity in entities)

    assert features[0] == features[1] == features[2] == features[3]


def test_primitive_options_instantiated_primitive(es):
    warning_msg = "Options present for primitive instance and generic " \
                  "primitive class \\(mean\\), primitive instance will not use generic " \
                  "options"

    skipna_mean = Mean(skipna=False)
    options = {
        skipna_mean: {'include_entities': ['stores']},
        'mean': {'ignore_entities': ['stores']}
    }
    with pytest.warns(UserWarning, match=warning_msg):
        dfs_obj = DeepFeatureSynthesis(target_entity_id='régions',
                                       entityset=es,
                                       agg_primitives=['mean', skipna_mean],
                                       trans_primitives=[],
                                       primitive_options=options)

    features = dfs_obj.build_features()
    for f in features:
        deps = f.get_dependencies(deep=True)
        entities = [d.entity.id for d in deps]
        if f.primitive == skipna_mean:
            assert all(entity == 'stores' for entity in entities)
        elif isinstance(f.primitive, Mean):
            assert 'stores' not in entities


def test_primitive_options_commutative(es):
    class AddThree(TransformPrimitive):
        name = 'add_three'
        input_types = [Numeric, Numeric, Numeric]
        return_type = Numeric
        commutative = True
        compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

        def generate_name(self, base_feature_names):
            return "%s + %s + %s" % (base_feature_names[0], base_feature_names[1], base_feature_names[2])

    options = {
        'add_numeric': [
            {'include_variables': {'log': ['value_2']}},
            {'include_variables': {'log': ['value']}}
        ],
        AddThree: [
            {'include_variables': {'log': ['value_2']}},
            {'include_variables': {'log': ['value_many_nans']}},
            {'include_variables': {'log': ['value']}}
        ]
    }
    dfs_obj = DeepFeatureSynthesis(target_entity_id='log',
                                   entityset=es,
                                   agg_primitives=[],
                                   trans_primitives=[AddNumeric, AddThree],
                                   primitive_options=options,
                                   max_depth=1)
    features = dfs_obj.build_features()
    add_numeric = [f for f in features if isinstance(f.primitive, AddNumeric)]
    assert len(add_numeric) == 1
    deps = add_numeric[0].get_dependencies(deep=True)
    assert deps[0].get_name() == 'value_2' and deps[1].get_name() == 'value'

    add_three = [f for f in features if isinstance(f.primitive, AddThree)]
    assert len(add_three) == 1
    deps = add_three[0].get_dependencies(deep=True)
    assert deps[0].get_name() == 'value_2' and deps[1].get_name() == 'value_many_nans' and deps[2].get_name() == 'value'


def test_primitive_ordering():
    # Test that the order of the input primitives impacts neither
    # which features are created nor their order
    es = make_ecommerce_entityset()

    trans_prims = [AddNumeric, Absolute, 'divide_numeric', NotEqual, 'is_null']
    groupby_trans_prim = ['cum_mean', CumMin, CumSum]
    agg_prims = [NMostCommon(n=3), Sum, Mean, Mean(skipna=False), 'min', 'max']
    where_prims = ['count', Sum]

    seed_num_chars = ft.Feature(es['customers']['favorite_quote'], primitive=NumCharacters)
    seed_is_null = ft.Feature(es['customers']['age'], primitive=IsNull)
    seed_features = [seed_num_chars, seed_is_null]

    features1 = ft.dfs(entityset=es,
                       target_entity="customers",
                       trans_primitives=trans_prims,
                       groupby_trans_primitives=groupby_trans_prim,
                       agg_primitives=agg_prims,
                       where_primitives=where_prims,
                       seed_features=seed_features,
                       max_features=-1,
                       max_depth=2,
                       features_only=2)

    trans_prims.reverse()
    groupby_trans_prim.reverse()
    agg_prims.reverse()
    where_prims.reverse()
    seed_features.reverse()

    features2 = ft.dfs(entityset=es,
                       target_entity="customers",
                       trans_primitives=trans_prims,
                       groupby_trans_primitives=groupby_trans_prim,
                       agg_primitives=agg_prims,
                       where_primitives=where_prims,
                       seed_features=seed_features,
                       max_features=-1,
                       max_depth=2,
                       features_only=2)

    assert len(features1) == len(features2)

    for i in range(len(features2)):
        assert features1[i].unique_name() == features2[i].unique_name()


def test_no_transform_stacking():
    df1 = pd.DataFrame({"id": [0, 1, 2, 3],
                        "A": [0, 1, 2, 3]})
    df2 = pd.DataFrame({'first_id': [0, 1, 1, 3], 'B': [99, 88, 77, 66]})

    entities = {"first": (df1, 'id'),
                "second": (df2, 'index')}
    relationships = [("first", 'id', 'second', 'first_id')]
    es = ft.EntitySet("data", entities, relationships)

    feature_defs = ft.dfs(entityset=es, target_entity='second',
                          trans_primitives=['negate', 'add_numeric'],
                          agg_primitives=['sum'],
                          max_depth=4,
                          features_only=2)
    expected = [
        'first_id',
        'B',
        '-(B)',
        'first.A',
        'first.SUM(second.B)',
        'first.-(A)',
        'B + first.A',
        'first.SUM(second.-(B))',
        'first.A + SUM(second.B)',
        'first.-(SUM(second.B))',
        'B + first.SUM(second.B)',
        'first.A + SUM(second.-(B))',
        'first.SUM(second.-(B)) + SUM(second.B)',
        'first.-(SUM(second.-(B)))',
        'B + first.SUM(second.-(B))'
    ]

    assert len(feature_defs) == len(expected)

    for feature_name in expected:
        assert feature_with_name(feature_defs, feature_name)
