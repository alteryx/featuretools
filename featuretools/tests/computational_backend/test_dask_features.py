from dask.base import tokenize

from ..testing_utils import make_ecommerce_entityset


def test_tokenize_entityset():
    es = make_ecommerce_entityset()
    dupe = make_ecommerce_entityset()
    int_es = make_ecommerce_entityset(with_integer_time_index=True)

    # check identitcal entitysets hash to same token
    assert tokenize(es) == tokenize(dupe)

    # not same if product relationship is missing
    productless = make_ecommerce_entityset()
    productless.relationships.pop()
    assert tokenize(es) != tokenize(productless)

    # not same if integer entityset
    assert tokenize(es) != tokenize(int_es)
