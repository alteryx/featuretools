from dask.base import tokenize

from ..testing_utils import make_ecommerce_entityset


def test_tokenize_entityset():
    es = make_ecommerce_entityset()
    dupe = make_ecommerce_entityset()

    # check identitcal entitysets hash to same token
    es_token = tokenize(es)
    assert es_token == tokenize(dupe)

    # not same if value in dataframe is changed
    no_ice = make_ecommerce_entityset()
    no_ice['customers'].df['loves_ice_cream'][0] = False
    assert tokenize(es['customers']) != tokenize(no_ice['customers'])
    assert es_token != tokenize(no_ice)

    # not same if product relationship is missing
    productless = make_ecommerce_entityset()
    productless.relationships.pop()
    assert es_token != tokenize(productless)

    # same if relationships are in different order
    scrambled = make_ecommerce_entityset()
    scrambled.relationships.reverse()
    assert es_token == tokenize(scrambled)
