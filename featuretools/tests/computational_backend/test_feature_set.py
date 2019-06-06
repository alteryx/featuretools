import featuretools as ft
from featuretools.computational_backends.feature_set import FeatureSet


def test_feature_trie(diamond_es):
    es = diamond_es
    r_to_country = _get_relationship(es, 'regions', 'countries')
    c_to_r = _get_relationship(es, 'customers', 'regions')
    t_to_c = _get_relationship(es, 'transactions', 'customers')
    s_to_r = _get_relationship(es, 'stores', 'regions')
    t_to_s = _get_relationship(es, 'transactions', 'stores')

    country_name = ft.IdentityFeature(es['countries']['name'])
    direct_name = ft.DirectFeature(country_name, es['regions'],
                                   relationship=r_to_country)
    amount = ft.IdentityFeature(es['transactions']['amount'])
    through_customers = ft.AggregationFeature(amount, es['regions'],
                                              primitive=ft.primitives.Mean,
                                              relationship_path=[c_to_r, t_to_c])
    through_stores = ft.AggregationFeature(amount, es['regions'],
                                           primitive=ft.primitives.Mean,
                                           relationship_path=[s_to_r, t_to_s])
    customers_mean = ft.AggregationFeature(amount, es['customers'],
                                           primitive=ft.primitives.Mean,
                                           relationship_path=[t_to_c])
    negation = ft.TransformFeature(customers_mean, ft.primitives.Negate)
    mean_of_mean = ft.AggregationFeature(negation, es['regions'],
                                         primitive=ft.primitives.Mean,
                                         relationship_path=[c_to_r])

    features = [direct_name, through_customers, through_stores, mean_of_mean]

    feature_set = FeatureSet(es, features)
    trie = feature_set.feature_trie

    assert trie.get_node([]).value == {f.unique_name() for f in features}
    assert trie.get_node([(True, r_to_country)]).value == {country_name.unique_name()}
    assert trie.get_node([(False, c_to_r)]).value == {negation.unique_name(), customers_mean.unique_name()}
    assert trie.get_node([(False, s_to_r)]).value == set()
    assert trie.get_node([(False, c_to_r), (False, t_to_c)]).value == {amount.unique_name()}
    assert trie.get_node([(False, s_to_r), (False, t_to_s)]).value == {amount.unique_name()}


def _get_relationship(es, child, parent):
    return next(r for r in es.relationships
                if r.child_entity.id == child and r.parent_entity.id == parent)
