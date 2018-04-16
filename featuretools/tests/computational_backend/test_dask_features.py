from dask.base import tokenize

from ..testing_utils import make_ecommerce_entityset


def test_tokenize_entityset():
    es_1 = make_ecommerce_entityset()
    es_2 = make_ecommerce_entityset()
    token_1 = tokenize(es_1)
    token_2 = tokenize(es_2)
    assert token_1 == token_2
