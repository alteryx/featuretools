import featuretools as ft
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.entityset.relationship import RelationshipPath
from featuretools.tests.testing_utils import backward_path
from featuretools.utils import Trie


def test_feature_trie_without_needs_full_entity(diamond_es):
    es = diamond_es
    country_name = ft.IdentityFeature(es['countries']['name'])
    direct_name = ft.DirectFeature(country_name, es['regions'])
    amount = ft.IdentityFeature(es['transactions']['amount'])

    path_through_customers = backward_path(es, ['regions', 'customers', 'transactions'])
    through_customers = ft.AggregationFeature(amount, es['regions'],
                                              primitive=ft.primitives.Mean,
                                              relationship_path=path_through_customers)
    path_through_stores = backward_path(es, ['regions', 'stores', 'transactions'])
    through_stores = ft.AggregationFeature(amount, es['regions'],
                                           primitive=ft.primitives.Mean,
                                           relationship_path=path_through_stores)
    customers_to_transactions = backward_path(es, ['customers', 'transactions'])
    customers_mean = ft.AggregationFeature(amount, es['customers'],
                                           primitive=ft.primitives.Mean,
                                           relationship_path=customers_to_transactions)

    negation = ft.TransformFeature(customers_mean, ft.primitives.Negate)
    regions_to_customers = backward_path(es, ['regions', 'customers'])
    mean_of_mean = ft.AggregationFeature(negation, es['regions'],
                                         primitive=ft.primitives.Mean,
                                         relationship_path=regions_to_customers)

    features = [direct_name, through_customers, through_stores, mean_of_mean]

    feature_set = FeatureSet(features)
    trie = feature_set.feature_trie

    assert trie.value == \
        (False, set(), {f.unique_name() for f in features})
    assert trie.get_node(direct_name.relationship_path).value == \
        (False, set(), {country_name.unique_name()})
    assert trie.get_node(regions_to_customers).value == \
        (False, set(), {negation.unique_name(), customers_mean.unique_name()})
    regions_to_stores = backward_path(es, ['regions', 'stores'])
    assert trie.get_node(regions_to_stores).value == (False, set(), set())
    assert trie.get_node(path_through_customers).value == \
        (False, set(), {amount.unique_name()})
    assert trie.get_node(path_through_stores).value == \
        (False, set(), {amount.unique_name()})


def test_feature_trie_with_needs_full_entity(diamond_es):
    pd_es = diamond_es
    amount = ft.IdentityFeature(pd_es['transactions']['amount'])

    path_through_customers = backward_path(pd_es, ['regions', 'customers', 'transactions'])
    agg = ft.AggregationFeature(amount, pd_es['regions'],
                                primitive=ft.primitives.Mean,
                                relationship_path=path_through_customers)
    trans_of_agg = ft.TransformFeature(agg, ft.primitives.CumSum)

    path_through_stores = backward_path(pd_es, ['regions', 'stores', 'transactions'])
    trans = ft.TransformFeature(amount, ft.primitives.CumSum)
    agg_of_trans = ft.AggregationFeature(trans, pd_es['regions'],
                                         primitive=ft.primitives.Mean,
                                         relationship_path=path_through_stores)

    features = [agg, trans_of_agg, agg_of_trans]
    feature_set = FeatureSet(features)
    trie = feature_set.feature_trie

    assert trie.value == \
        (True, {agg.unique_name(), trans_of_agg.unique_name()}, {agg_of_trans.unique_name()})
    assert trie.get_node(path_through_customers).value == \
        (True, {amount.unique_name()}, set())
    assert trie.get_node(path_through_customers[:1]).value == (True, set(), set())
    assert trie.get_node(path_through_stores).value == \
        (True, {amount.unique_name(), trans.unique_name()}, set())
    assert trie.get_node(path_through_stores[:1]).value == (False, set(), set())


def test_feature_trie_with_needs_full_entity_direct(es):
    value = ft.IdentityFeature(es['log']['value'],)
    agg = ft.AggregationFeature(value, es['sessions'],
                                primitive=ft.primitives.Mean)
    agg_of_agg = ft.AggregationFeature(agg, es['customers'],
                                       primitive=ft.primitives.Sum)
    direct = ft.DirectFeature(agg_of_agg, es['sessions'])
    trans = ft.TransformFeature(direct, ft.primitives.CumSum)

    features = [trans, agg]
    feature_set = FeatureSet(features)
    trie = feature_set.feature_trie

    assert trie.value == \
        (True, {direct.unique_name(), trans.unique_name()}, {agg.unique_name()})

    assert trie.get_node(agg.relationship_path).value == \
        (False, set(), {value.unique_name()})

    parent_node = trie.get_node(direct.relationship_path)
    assert parent_node.value == (True, {agg_of_agg.unique_name()}, set())

    child_through_parent_node = parent_node.get_node(agg_of_agg.relationship_path)
    assert child_through_parent_node.value == (True, {agg.unique_name()}, set())

    assert child_through_parent_node.get_node(agg.relationship_path).value == \
        (True, {value.unique_name()}, set())


def test_feature_trie_ignores_approximate_features(es):
    value = ft.IdentityFeature(es['log']['value'],)
    agg = ft.AggregationFeature(value, es['sessions'],
                                primitive=ft.primitives.Mean)
    agg_of_agg = ft.AggregationFeature(agg, es['customers'],
                                       primitive=ft.primitives.Sum)
    direct = ft.DirectFeature(agg_of_agg, es['sessions'])
    features = [direct, agg]

    approximate_feature_trie = Trie(default=list, path_constructor=RelationshipPath)
    approximate_feature_trie.get_node(direct.relationship_path).value = [agg_of_agg]
    feature_set = FeatureSet(features, approximate_feature_trie=approximate_feature_trie)
    trie = feature_set.feature_trie

    # Since agg_of_agg is ignored it and its dependencies should not be in the
    # trie.
    sub_trie = trie.get_node(direct.relationship_path)
    for _path, (_, _, features) in sub_trie:
        assert not features

    assert trie.value == (False, set(), {direct.unique_name(), agg.unique_name()})
    assert trie.get_node(agg.relationship_path).value == \
        (False, set(), {value.unique_name()})
