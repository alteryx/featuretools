import featuretools as ft
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.tests.testing_utils import backward_path


def test_feature_trie(diamond_es):
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

    feature_set = FeatureSet(es, features)
    trie = feature_set.feature_trie

    assert trie.value == {f.unique_name() for f in features}
    assert trie.get_node(direct_name.relationship_path).value == {country_name.unique_name()}
    assert trie.get_node(regions_to_customers).value == {negation.unique_name(), customers_mean.unique_name()}
    regions_to_stores = backward_path(es, ['regions', 'stores'])
    assert trie.get_node(regions_to_stores).value == set()
    assert trie.get_node(path_through_customers).value == {amount.unique_name()}
    assert trie.get_node(path_through_stores).value == {amount.unique_name()}
